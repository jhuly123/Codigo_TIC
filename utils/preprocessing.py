# preprocessing.py — normalización e interpolación de keypoints MediaPipe,
# aplicadas a cada secuencia antes de entrenar o evaluar el modelo.

import numpy as np
from .config import N_FRAMES, N_KEYPOINTS


def normalizar_hombros(secuencia: np.ndarray) -> np.ndarray:
    """
    Centra cada frame en el punto medio entre hombros (landmarks 11-12 de
    MediaPipe Pose) y escala por la distancia inter-hombros.
    Aplica la misma transformación a las manos para mantener coherencia.

    Entrada/salida: ndarray (T, 225).
    Si la distancia inter-hombros es < 0.001 (no detectada), escala = 1.
    """
    r = secuencia.copy()
    for i, frame in enumerate(secuencia):
        pose   = frame[:99].reshape(33, 3)
        centro = (pose[11] + pose[12]) / 2
        escala = np.linalg.norm(pose[12] - pose[11])
        if escala < 0.001:
            escala = 1.0
        r[i, :99] = ((pose - centro) / escala).flatten()
        for ini, fin in [(99, 162), (162, 225)]:
            pts = frame[ini:fin].reshape(21, 3)
            if np.any(pts != 0):
                r[i, ini:fin] = ((pts - centro) / escala).flatten()
    return r


def interpolar(secuencia: np.ndarray, n_frames: int = N_FRAMES) -> np.ndarray:
    """
    Remuestrea la secuencia a exactamente n_frames usando selección de
    índices enteros (sin suavizado). Maneja secuencias vacías con ceros.
    """
    n = len(secuencia)
    if n == 0:
        return np.zeros((n_frames, N_KEYPOINTS))
    return secuencia[np.linspace(0, n - 1, n_frames).astype(int)]


def extraer_kp_frame(results) -> np.ndarray:
    """
    Concatena los 225 keypoints de un frame MediaPipe Holistic.
    Retorna ceros para las partes del cuerpo no detectadas, garantizando
    un vector de longitud fija independientemente de la detección.

    Nota de diseño: los landmarks faciales (468 pts del Face Mesh) no se
    incluyen deliberadamente. Con el corpus actual (~4 muestras/clase) la
    incorporación de dimensiones faciales agrava la maldición de la
    dimensionalidad sin aporte discriminativo neto. Esta decisión debe
    revisarse cuando el corpus alcance >= 50 muestras por clase.
    """
    pose = (np.array([[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(99))
    lh   = (np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(63))
    rh   = (np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(63))
    return np.concatenate([pose, lh, rh])
