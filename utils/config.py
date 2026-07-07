# config.py — único punto de configuración del pipeline.
# Todas las rutas se derivan de LSEC_DATA_DIR para que el proyecto sea
# portable entre máquinas.

import os
from pathlib import Path

_PROYECTO = str(Path(__file__).parent.parent)
DATA_DIR = os.getenv("LSEC_DATA_DIR", _PROYECTO)

# Directorios de keypoints
KEYPOINTS_DIR  = os.path.join(DATA_DIR, "keypoints", "label_studio")
KP_CVAT_DIR    = os.path.join(DATA_DIR, "keypoints", "cvat")
KP_ELAN_DIR    = os.path.join(DATA_DIR, "keypoints", "elan")
KP_AUG_DIR     = os.path.join(DATA_DIR, "keypoints_aumentado_v2")
DATASET_V2_DIR = os.path.join(DATA_DIR, "dataset_final_v2")

# Split fijo, compartido por todos los experimentos
EXP_DIR    = os.path.join(DATA_DIR, "experimentos", "exp1_ab")
SPLIT_PATH = os.path.join(EXP_DIR,  "split_fijo.npz")

# CSVs del pipeline
_DATOS        = os.path.join(DATA_DIR, "datos")
CSV_TOP35     = os.path.join(_DATOS, "dataset_top35.csv")
CSV_LIMPIO    = os.path.join(DATA_DIR, "anotaciones", "label_studio", "dataset_limpio.csv")
CSV_KEYPOINTS = os.path.join(_DATOS, "dataset_keypoints.csv")
CSV_AUG_V2    = os.path.join(_DATOS, "dataset_aumentado_v2.csv")
CSV_COMP35    = os.path.join(_DATOS, "dataset_comparativa_35.csv")

# Anotaciones originales de CVAT/ELAN: primero busca dentro del propio
# proyecto; si no existen, cae a ~/etiquetado. Override con LSEC_CVAT_DIR /
# LSEC_ELAN_DIR.
_USER_HOME   = str(Path.home())
_ETIQUETADO  = os.path.join(_USER_HOME, "etiquetado")
_cvat_local  = os.path.join(DATA_DIR, "anotaciones", "cvat")
_elan_local  = os.path.join(DATA_DIR, "anotaciones", "elan")
CVAT_ANOTACIONES_DIR = os.getenv("LSEC_CVAT_DIR",
    _cvat_local if os.path.isdir(_cvat_local)
    else os.path.join(_ETIQUETADO, "anotaciones", "cvat"))
ELAN_ANOTACIONES_DIR = os.getenv("LSEC_ELAN_DIR",
    _elan_local if os.path.isdir(_elan_local)
    else os.path.join(_ETIQUETADO, "anotaciones", "elan"))

N_FRAMES    = 60    # frames por secuencia, tras interpolación temporal
N_KEYPOINTS = 225   # pose(33x3=99) + mano_izq(21x3=63) + mano_der(21x3=63)
