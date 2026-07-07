"""
Prototipo pequeño de traductor en tiempo real.

Controles:
  -> / D   siguiente sena objetivo (modo guiado)
  <- / A   sena objetivo anterior  (modo guiado)
  M        alternar modo guiado / libre
  R        reiniciar buffer
  S        captura
  Q        salir

"""

import os, sys, cv2, argparse
import numpy as np
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
from collections import deque
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_DIR, N_FRAMES, N_KEYPOINTS, CSV_TOP35
from utils.preprocessing import normalizar_hombros, extraer_kp_frame

MODELO_PATH = os.path.join(DATA_DIR, "modelo", "resultados", "mejor_modelo_v2.keras")

# Senas objetivo del modo guiado, ordenadas por precision real del modelo v2b
# determinista (modelo/resultados/reporte_por_clase.csv). El numero es la
# precision: cuando el modelo predice esta sena, que tan seguido acierta.
SENAS_OBJETIVO = [
    ("CORRECTO",    100),
    ("CRECER",      100),
    ("ESCUCHAR",    100),
    ("ACOMODAR",    100),
    ("APROBAR",     100),
    ("ENOJADO",      50),
    ("ACCIDENTE",    50),
    ("FAMOSO",       50),
    ("MUDARSE",      33),
    ("AVERGONZAR",   20),
]

C_VERDE   = (80,  200,  80)
C_ROJO    = (60,   60, 220)
C_ACENTO  = (255, 180,  30)
C_BLANCO  = (255, 255, 255)
C_GRIS    = (150, 150, 150)
FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def cargar_clases():
    import pandas as pd
    return sorted(pd.read_csv(CSV_TOP35)['glosa'].unique())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camara', type=int, default=0)
    parser.add_argument('--umbral', type=float, default=0.5)
    args = parser.parse_args()

    print("Cargando modelo...", end=' ', flush=True)
    model  = tf.keras.models.load_model(MODELO_PATH)
    clases = cargar_clases()
    n_cl   = len(clases)
    print(f"OK - {n_cl} senas")

    # Senas del modo guiado presentes en las clases del modelo
    objetivo = [(g, p) for g, p in SENAS_OBJETIVO if g in clases]
    idx_permitidos = np.array([clases.index(g) for g, _ in objetivo])

    cap = None
    for idx in ([args.camara] if args.camara != 0 else [0, 1, 2]):
        c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if c.isOpened() and c.read()[0]:
            cap = c
            print(f"Camara {idx} OK")
            break
        c.release()
    if cap is None:
        print("Error: no se encontro camara")
        return

    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    buffer      = deque(maxlen=N_FRAMES)
    suavizado   = deque(maxlen=4)
    pred        = ""
    conf        = 0.0
    umbral      = args.umbral
    sin_manos   = 0
    modo_guiado = bool(objetivo)
    idx_sena    = 0

    def reset_prediccion():
        nonlocal pred, conf
        buffer.clear(); suavizado.clear()
        pred = ""; conf = 0.0

    capturas_dir = os.path.join(DATA_DIR, "demo_capturas",
                                datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(capturas_dir, exist_ok=True)

    print("Iniciado. Q=salir  R=reiniciar  S=captura  M=modo  A/D=cambiar sena")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Los keypoints se extraen del frame SIN espejar: el modelo se entrenó
        # con videos sin espejo y cv2.flip invertiría la lateralidad de manos.
        # El espejo se aplica solo a la visualización, más abajo.
        h, w      = frame.shape[:2]
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results    = holistic.process(rgb)
        manos_ok   = (results.left_hand_landmarks is not None or
                      results.right_hand_landmarks is not None)

        mpd = mp.solutions.drawing_utils
        mpd.draw_landmarks(frame, results.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mpd.DrawingSpec((100,100,100), 1, 2))
        if results.left_hand_landmarks:
            mpd.draw_landmarks(frame, results.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mpd.DrawingSpec((50, 200, 50), 2, 3),
                connection_drawing_spec=mpd.DrawingSpec((30, 150, 30), 2))
        if results.right_hand_landmarks:
            mpd.draw_landmarks(frame, results.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mpd.DrawingSpec((50, 100, 255), 2, 3),
                connection_drawing_spec=mpd.DrawingSpec((20, 70, 200), 2))

        # Buffer solo con manos
        if manos_ok:
            sin_manos = 0
            kp = extraer_kp_frame(results)
            if kp.shape[0] == N_KEYPOINTS:
                buffer.append(kp)
        else:
            sin_manos += 1
            if sin_manos > 20:
                reset_prediccion()

        if len(buffer) == N_FRAMES:
            seq = normalizar_hombros(np.array(buffer, dtype=np.float32))
            p   = model.predict(seq[np.newaxis], verbose=0)[0]
            suavizado.append(p)
            ps = np.mean(suavizado, axis=0)
            if modo_guiado:
                # Restringe la prediccion a las senas objetivo
                # (no compite contra las 35 clases completas).
                idx  = int(idx_permitidos[np.argmax(ps[idx_permitidos])])
                conf = float(ps[idx] / ps[idx_permitidos].sum())
            else:
                idx  = int(np.argmax(ps))
                conf = float(ps[idx])
            pred = clases[idx]

        # --- UI (sobre el frame espejado, efecto espejo natural) ---
        frame = cv2.flip(frame, 1)

        # Banda superior: sena objetivo (solo en modo guiado)
        if modo_guiado:
            glosa_obj, prec_obj = objetivo[idx_sena]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 52), (10, 10, 15), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"({idx_sena+1}/{len(objetivo)})  A/D cambiar  M modo libre",
                        (10, 18), FONT, 0.42, C_GRIS, 1, cv2.LINE_AA)
            cv2.putText(frame, "Haz la sena:", (10, 42), FONT, 0.55, C_BLANCO, 1, cv2.LINE_AA)
            esc = min(1.0, 10.0 / max(len(glosa_obj), 1))
            cv2.putText(frame, glosa_obj, (130, 42), FONT_BOLD, esc, C_ACENTO, 2, cv2.LINE_AA)
            cv2.putText(frame, f"prec. modelo {prec_obj}%",
                        (w - 165, 18), FONT, 0.42, C_ACENTO, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, h-70), (w, h), (0,0,0), -1)

        if pred and conf >= umbral:
            if modo_guiado:
                correcto = (pred == objetivo[idx_sena][0])
                color    = C_VERDE if correcto else C_ROJO
                texto    = f"{pred}  ({conf*100:.0f}%)"
                if correcto:
                    texto += "   CORRECTO!"
            else:
                color = (0, 220, 220)
                texto = f"{pred}  ({conf*100:.0f}%)"
            cv2.putText(frame, texto, (15, h-38), FONT_BOLD, 0.9, color, 2)
        else:
            cv2.putText(frame, "Esperando...", (15, h-38),
                        FONT, 0.7, C_GRIS, 1)

        # Buffer y estado
        estado = "Manos: SI" if manos_ok else "Manos: NO"
        modo   = "GUIADO" if modo_guiado else "LIBRE"
        cv2.putText(frame, f"Buffer: {len(buffer)}/{N_FRAMES}   {estado}   Modo: {modo}",
                    (15, h-12), FONT, 0.45, (180,180,180), 1)

        bx = w - 220
        cv2.rectangle(frame, (bx, h-18), (bx+200, h-8), (60,60,60), -1)
        fill = int(200 * len(buffer) / N_FRAMES)
        cv2.rectangle(frame, (bx, h-18), (bx+fill, h-8), (50,200,50), -1)

        cv2.imshow("Demo LSEc", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_prediccion()
        elif key == ord('m') and objetivo:
            modo_guiado = not modo_guiado
            reset_prediccion()
            print(f"  [M] Modo: {'GUIADO' if modo_guiado else 'LIBRE'}")
        elif modo_guiado and key in (ord('d'), ord('.'), 83):   # siguiente
            idx_sena = (idx_sena + 1) % len(objetivo)
            reset_prediccion()
            print(f"  -> {objetivo[idx_sena][0]}")
        elif modo_guiado and key in (ord('a'), ord(','), 81):   # anterior
            idx_sena = (idx_sena - 1) % len(objetivo)
            reset_prediccion()
            print(f"  <- {objetivo[idx_sena][0]}")
        elif key == ord('s'):
            nombre = os.path.join(capturas_dir,
                f"{pred or 'sin'}_{int(conf*100)}pct_{datetime.now().strftime('%H%M%S')}.png")
            cv2.imwrite(nombre, frame)
            print(f"Captura: {os.path.basename(nombre)}")

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
