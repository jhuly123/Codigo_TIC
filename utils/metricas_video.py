# metricas_video.py — análisis de segmentos de video con MediaPipe Holistic,
# usado por pipeline/04_extraer_metadatos.py al procesar cada tarea de
# Label Studio.
#
# Para procesar muchos videos seguidos, inicializa Holistic UNA SOLA VEZ
# fuera del bucle y pásalo en `holistic` (si se deja en None, se crea y
# destruye por llamada — conveniente pero lento en bucles grandes):
#
#   with mp.solutions.holistic.Holistic(...) as h:
#       for video_path, t_ini, t_fin in segmentos:
#           metricas = analizar_segmento(video_path, t_ini, t_fin, holistic=h)

import os
import cv2
import numpy as np
import mediapipe as mp


def resolver_video_path(video_id: str, glosa: str, videos_root: str) -> str | None:
    """
    Localiza el archivo de video a partir del video_id y la glosa.
    Primero intenta videos/{GLOSA}/{video_id}.mp4 (ruta directa); si no
    existe, busca en todas las subcarpetas (fallback para glosas renombradas).
    """
    nombre = video_id + ".mp4"

    candidato = os.path.join(videos_root, glosa, nombre)
    if os.path.exists(candidato):
        return candidato

    if os.path.isdir(videos_root):
        for carpeta in os.listdir(videos_root):
            candidato = os.path.join(videos_root, carpeta, nombre)
            if os.path.exists(candidato):
                return candidato

    return None


def analizar_segmento(video_path: str,
                      t_ini: float,
                      t_fin: float,
                      holistic=None) -> dict | None:
    """
    Analiza un segmento de video y calcula métricas de calidad con MediaPipe:
    fps, keypoints_validos (fracción de frames con al menos una mano
    detectada), mano_dominante, blur_score (varianza del Laplaciano) y
    velocidad_media de las muñecas.

    Retorna None si el video no se puede abrir o el segmento está fuera de rango.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_ini = int(t_ini * fps)
    frame_fin = int(t_fin * fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_fin    = min(frame_fin, total_frames - 1)
    if frame_ini > frame_fin:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)

    _holistic_propio = holistic is None
    if _holistic_propio:
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    blur_scores  = []
    kp_frames    = []
    vel_der_list = []
    vel_izq_list = []
    vel_list     = []
    pos_prev_der = pos_prev_izq = pos_prev = None

    frame_idx = frame_ini
    while frame_idx <= frame_fin:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        pos_der = pos_izq = pos_act = None
        manos   = 0

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if lm[16].visibility > 0.5:
                pos_der = np.array([lm[16].x, lm[16].y])
                manos  += 1
            if lm[15].visibility > 0.5:
                pos_izq = np.array([lm[15].x, lm[15].y])
                manos  += 1
            if pos_der is not None and pos_izq is not None:
                pos_act = (pos_der + pos_izq) / 2
            elif pos_der is not None:
                pos_act = pos_der
            elif pos_izq is not None:
                pos_act = pos_izq

        # Landmarks de manos directos (más precisos que pose)
        if results.right_hand_landmarks:
            pos_der = np.array([results.right_hand_landmarks.landmark[0].x,
                                results.right_hand_landmarks.landmark[0].y])
            manos = max(manos, 1)
        if results.left_hand_landmarks:
            pos_izq = np.array([results.left_hand_landmarks.landmark[0].x,
                                results.left_hand_landmarks.landmark[0].y])
            manos = max(manos, 1)

        kp_frames.append(1 if manos > 0 else 0)

        # Velocidades
        v     = float(np.linalg.norm(pos_act - pos_prev)) if (pos_act is not None and pos_prev is not None) else 0.0
        v_der = float(np.linalg.norm(pos_der - pos_prev_der)) if (pos_der is not None and pos_prev_der is not None) else 0.0
        v_izq = float(np.linalg.norm(pos_izq - pos_prev_izq)) if (pos_izq is not None and pos_prev_izq is not None) else 0.0

        vel_list.append(v)
        vel_der_list.append(v_der)
        vel_izq_list.append(v_izq)

        pos_prev     = pos_act
        pos_prev_der = pos_der
        pos_prev_izq = pos_izq
        frame_idx   += 1

    cap.release()
    if _holistic_propio:
        holistic.close()

    if not kp_frames:
        return None

    keypoints_validos = round(sum(kp_frames) / len(kp_frames), 3)
    blur_score        = round(float(np.mean(blur_scores)), 2) if blur_scores else 0.0
    velocidad_media   = round(float(np.mean(vel_list)), 5) if vel_list else 0.0

    # Mano dominante (umbral 30%: ratio > 1.3 → derecha, < 0.7 → izquierda)
    vd_total = sum(vel_der_list)
    vi_total = sum(vel_izq_list)
    umbral   = 0.3

    if vd_total == 0 and vi_total == 0:
        mano_dominante = "ninguna"
    elif vd_total == 0:
        mano_dominante = "izquierda"
    elif vi_total == 0:
        mano_dominante = "derecha"
    else:
        ratio = vd_total / max(vi_total, 0.0001)
        if ratio > (1 + umbral):
            mano_dominante = "derecha"
        elif ratio < (1 - umbral):
            mano_dominante = "izquierda"
        else:
            mano_dominante = "ambas_der" if vd_total >= vi_total else "ambas_izq"

    return {
        "fps":               round(fps, 2),
        "keypoints_validos": keypoints_validos,
        "mano_dominante":    mano_dominante,
        "blur_score":        blur_score,
        "velocidad_media":   velocidad_media,
    }
