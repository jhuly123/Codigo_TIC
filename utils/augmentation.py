# augmentation.py — primitivas de aumento de datos. Ninguna modifica la
# secuencia de entrada: todas retornan un array nuevo.
#
# pipeline/09_aumentar_datos.py usa TECNICAS_V2B (producción, x13 el train
# set); experimentos/exp3_tecnicas/generar_aumento_15tecnicas.py usa
# TECNICAS_V2 (comparativa v1=9 / v2b=12 / v2=15 del Experimento 3).

import numpy as np
from scipy.interpolate import interp1d
from .config import N_FRAMES, N_KEYPOINTS
from .preprocessing import interpolar

# Pares de landmarks simétricos de MediaPipe Pose (para swap en flip)
_PARES_SIM = [(11,12),(13,14),(15,16),(17,18),(19,20),
              (21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]


def flip_seq(seq: np.ndarray) -> np.ndarray:
    """Espejo horizontal con swap correcto de landmarks izquierda↔derecha."""
    f = seq.copy()
    for i, frame in enumerate(f):
        p  = frame[:99].reshape(33, 3)
        # .copy() obligatorio: lh/rh serían vistas sobre f y el swap de abajo
        # sobrescribiría la mano izquierda antes de poder leerla.
        lh = frame[99:162].reshape(21, 3).copy()
        rh = frame[162:225].reshape(21, 3).copy()
        p[:,0] *= -1;  lh[:,0] *= -1;  rh[:,0] *= -1
        for a, b in _PARES_SIM:
            p[[a, b]] = p[[b, a]]
        f[i, :99]     = p.flatten()
        f[i, 99:162]  = rh.flatten()   # swap izq↔der
        f[i, 162:225] = lh.flatten()
    return f


def warp_seq(seq: np.ndarray, factor: float) -> np.ndarray:
    """
    Estira o comprime la secuencia temporalmente por factor y reinterprola
    a N_FRAMES. factor < 1 = movimiento lento; factor > 1 = movimiento rápido.
    """
    n     = len(seq)
    x_o   = np.linspace(0, n - 1, n)
    x_new = np.linspace(0, n - 1, max(1, int(n * factor)))
    w = interp1d(x_o, seq, axis=0, kind='linear', fill_value='extrapolate')(x_new)
    return interpolar(w)


def noise_seq(seq: np.ndarray, sigma: float = 0.008) -> np.ndarray:
    """Ruido gaussiano (σ=sigma) aplicado únicamente a keypoints ≠ 0."""
    r = seq.copy()
    mask = seq != 0
    r[mask] += np.random.normal(0, sigma, seq.shape)[mask]
    return r


def rotate_seq(seq: np.ndarray, deg: float) -> np.ndarray:
    """Rotación 2D en el plano XY aplicada a pose y ambas manos."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    r = seq.copy()
    for i, frame in enumerate(seq):
        for ini, fin in [(0, 99), (99, 162), (162, 225)]:
            pts = frame[ini:fin].reshape(-1, 3).copy()
            if np.any(pts != 0):
                pts[:, 0], pts[:, 1] = (pts[:, 0]*c - pts[:, 1]*s,
                                         pts[:, 0]*s + pts[:, 1]*c)
                r[i, ini:fin] = pts.flatten()
    return r


def scale_seq(seq: np.ndarray, factor: float = None) -> np.ndarray:
    """Escala global de la secuencia. Sin factor, uno aleatorio en [0.85, 1.15]."""
    if factor is None:
        factor = np.random.uniform(0.85, 1.15)
    return seq * factor


def translate_seq(seq: np.ndarray) -> np.ndarray:
    """Desplaza x,y de los keypoints detectados por un offset aleatorio."""
    r = seq.copy()
    dx, dy = np.random.uniform(-0.05, 0.05, size=2)
    for i, frame in enumerate(seq):
        for ini, fin in [(0, 99), (99, 162), (162, 225)]:
            pts = frame[ini:fin].reshape(-1, 3).copy()
            if np.any(pts != 0):
                m = np.any(pts != 0, axis=1)
                pts[m, 0] += dx
                pts[m, 1] += dy
                r[i, ini:fin] = pts.flatten()
    return r


def frame_drop_seq(seq: np.ndarray, prop: float = 0.15) -> np.ndarray:
    """Elimina ~prop de los frames al azar y reinterpola a N_FRAMES."""
    n = len(seq)
    n_keep = max(2, int(n * (1 - prop)))
    idx = np.sort(np.random.choice(n, n_keep, replace=False))
    return interpolar(seq[idx])


def aumentar_secuencia(seq: np.ndarray) -> list:
    """
    Genera 5 variantes aumentadas de una secuencia ya normalizada.
    Retorna lista de 6 arrays (60, 225): [original, flip, slow, fast, noise, rot+8°].

    Estas 5 técnicas son las usadas en los experimentos A/B y factorial 2×2.
    Para el dataset de producción (12 técnicas) ver TECNICAS_V2B.
    """
    return [
        seq,
        flip_seq(seq),
        warp_seq(seq, 0.75),
        warp_seq(seq, 1.25),
        noise_seq(seq),
        rotate_seq(seq, 8),
    ]


# Registro de técnicas por versión (comparadas en experimentos/exp3_tecnicas)
# v1 (9)  = geométricas + estocásticas + combinadas base.
# v2b (12) = v1 + translate/fast_flip/rot_noise (sin temporales extremas).
# v2  (15) = v2b + very_slow/very_fast/frame_drop (temporales extremas).
#
# exp3_tecnicas_temporales.ipynb concluyó que v2b generaliza mejor a CVAT/ELAN
# que v2 (las temporales extremas degradan la robustez entre herramientas) —
# ver experimentos/exp3_tecnicas/resultados_tecnicas_temporales.csv. Por eso
# PRODUCCIÓN usa TECNICAS_V2B (pipeline/09_aumentar_datos.py, ×13).
# TECNICAS_V2 sigue existiendo para reproducir exp3
# (experimentos/exp3_tecnicas/generar_aumento_15tecnicas.py).
TECNICAS_V1 = {
    'flip':       flip_seq,
    'slow':       lambda s: warp_seq(s, 0.75),
    'fast':       lambda s: warp_seq(s, 1.25),
    'noise':      lambda s: noise_seq(s, sigma=0.01),
    'rot_pos':    lambda s: rotate_seq(s,  10),
    'rot_neg':    lambda s: rotate_seq(s, -10),
    'scale':      scale_seq,
    'flip_noise': lambda s: noise_seq(flip_seq(s), sigma=0.01),
    'slow_noise': lambda s: noise_seq(warp_seq(s, 0.75), sigma=0.01),
}

TECNICAS_V2B = {
    **TECNICAS_V1,
    'translate':  translate_seq,
    'fast_flip':  lambda s: warp_seq(flip_seq(s), 1.25),
    'rot_noise':  lambda s: noise_seq(rotate_seq(s, 10), sigma=0.01),
}

TECNICAS_V2 = {
    **TECNICAS_V2B,
    'very_slow':  lambda s: warp_seq(s, 0.5),
    'very_fast':  lambda s: warp_seq(s, 1.5),
    'frame_drop': frame_drop_seq,
}
