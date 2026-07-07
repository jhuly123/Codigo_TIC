# experiment_utils.py — funciones compartidas por los notebooks
# exp1_ab_5fold.ipynb, exp2_factorial_5fold.ipynb y exp_ablacion_pipeline.ipynb
# (antes copiadas y pegadas en cada uno).

import os
import random
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def fijar_semillas(seed: int):
    """Fija todas las semillas (hash, random, numpy, tensorflow) para reproducibilidad."""
    import tensorflow as tf   # import diferido: los scripts fijan env vars antes de importar TF
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def wilson_ci(aciertos: int, total: int, z: float = 1.96):
    """Intervalo de confianza Wilson al 95% para una proporción."""
    if total == 0:
        return 0.0, 0.0
    p      = aciertos / total
    denom  = 1 + z ** 2 / total
    centro = (p + z ** 2 / (2 * total)) / denom
    margen = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return max(0.0, round(centro - margen, 4)), min(1.0, round(centro + margen, 4))


def dividir_train_val(X, y, seed: int, val_ratio: float = 0.15):
    """
    Divide train/val estratificado con semilla fija.
    Fallback a ShuffleSplit simple cuando el val set sería menor que el
    número de clases (StratifiedShuffleSplit lanza ValueError en ese caso).
    """
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        idx_tr, idx_val = next(sss.split(X, y))
    except ValueError:
        ss = ShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        idx_tr, idx_val = next(ss.split(X, y))
    return X[idx_tr], y[idx_tr], X[idx_val], y[idx_val]
