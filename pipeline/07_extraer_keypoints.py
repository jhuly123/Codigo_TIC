"""
Función: extrae keypoints MediaPipe para todas las herramientas de anotación
usando dataset_limpio.csv. Guarda los .npy en keypoints/{herramienta}/.
"""

import os
import sys
import ctypes
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import DATA_DIR, N_FRAMES, CSV_LIMPIO, KEYPOINTS_DIR
from utils.preprocessing import normalizar_hombros, interpolar, extraer_kp_frame
from utils.metricas_video import resolver_video_path

KP_BASE_DIR  = os.path.join(DATA_DIR, "keypoints")
VIDEOS_ROOT  = os.path.join(DATA_DIR, "videos")


def ruta_corta(path: str) -> str:
    try:
        buf = ctypes.create_unicode_buffer(32768)
        ctypes.windll.kernel32.GetShortPathNameW(path, buf, 32768)
        return buf.value or path
    except Exception:
        return path


def procesar_video(video_path: str, t_ini: float, t_fin: float, holistic) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(ruta_corta(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0   # mismo fallback que utils/metricas_video
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_ini * fps))

    kps, f = [], int(t_ini * fps)
    while f <= int(t_fin * fps):
        ret, frame = cap.read()
        if not ret:
            break
        kps.append(extraer_kp_frame(
            holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        f += 1
    cap.release()

    if not kps:
        return None
    return normalizar_hombros(interpolar(np.array(kps), N_FRAMES)).astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Extrae keypoints MediaPipe (todas las herramientas)")
    ap.add_argument("--solo", default=None,
                    help="Procesar solo esta herramienta (ej: --solo cvat). "
                         "Por defecto procesa label_studio, cvat y elan.")
    ap.add_argument("--skip-existentes", action="store_true",
                    help="Omitir videos cuyo .npy ya existe en disco.")
    args = ap.parse_args()

    if not os.path.exists(CSV_LIMPIO):
        print(f"[ERROR] No existe {CSV_LIMPIO}")
        print("  Ejecuta primero el notebook pipeline/06_limpiar_dataset.ipynb")
        return

    df           = pd.read_csv(CSV_LIMPIO)
    herramientas = df["herramienta"].unique()
    if args.solo:
        if args.solo not in herramientas:
            print(f"[ERROR] Herramienta '{args.solo}' no encontrada. "
                  f"Disponibles: {list(herramientas)}")
            return
        herramientas = [args.solo]
        print(f"Modo --solo: procesando únicamente '{args.solo}'")
    print(f"Dataset limpio: {len(df)} registros | herramientas: {list(herramientas)}")

    dirs_salida = {}
    for h in herramientas:
        # Cada herramienta escribe en keypoints/{herramienta}/. Para label_studio
        # eso es KEYPOINTS_DIR (keypoints/label_studio/), de donde leen los pasos
        # 09 y 10 y todos los experimentos (ver utils/config.py).
        if h == "label_studio":
            d = KEYPOINTS_DIR
        else:
            d = os.path.join(KP_BASE_DIR, h)
        os.makedirs(d, exist_ok=True)
        dirs_salida[h] = d

    resumen = {h: {"ok": 0, "err": 0} for h in herramientas}

    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        for h in herramientas:
            filas = df[df["herramienta"] == h]
            print(f"\n{'='*50}\n{h.upper()} — {len(filas)} videos\n{'='*50}")

            for _, row in filas.iterrows():
                video_id = str(row["video_id"])
                glosa    = str(row["glosa"])
                dst_path = os.path.join(dirs_salida[h], f"{video_id}.npy")

                # usa tiempo_inicio/tiempo_fin de cada herramienta;
                # video_path puede estar vacío en registros CVAT/ELAN
                video_path = None
                vp_csv = row.get("video_path")
                if pd.notna(vp_csv) and vp_csv != "" and os.path.exists(str(vp_csv)):
                    video_path = str(vp_csv)
                if video_path is None:
                    video_path = resolver_video_path(video_id, glosa, VIDEOS_ROOT)

                if video_path is None:
                    print(f"  [SKIP] {video_id} — video no encontrado")
                    resumen[h]["err"] += 1
                    continue

                if args.skip_existentes and os.path.exists(dst_path):
                    print(f"  [SKIP] {video_id} — .npy ya existe")
                    resumen[h]["ok"] += 1
                    continue

                print(f"  {glosa}...", end=" ", flush=True)
                kp = procesar_video(video_path,
                                    float(row["tiempo_inicio"]),
                                    float(row["tiempo_fin"]), holistic)
                if kp is None:
                    print("ERROR al procesar video")
                    resumen[h]["err"] += 1
                    continue

                np.save(dst_path, kp)
                print(f"OK {kp.shape}")
                resumen[h]["ok"] += 1

    print(f"\n{'='*50}\nRESUMEN FINAL")
    print(f"{'Herramienta':<20} {'OK':>6} {'Errores':>8}")
    print("-" * 36)
    for h, r in resumen.items():
        print(f"{h:<20} {r['ok']:>6} {r['err']:>8}")
    print(f"\nKeypoints en: {KP_BASE_DIR}/{{herramienta}}/")


if __name__ == "__main__":
    main()
