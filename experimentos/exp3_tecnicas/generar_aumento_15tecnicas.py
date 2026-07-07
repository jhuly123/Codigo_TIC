"""
experimentos/exp3_tecnicas/generar_aumento_15tecnicas.py
=========================================================
Genera una copia AISLADA del dataset de aumento con las 15 técnicas, usada por
el Experimento 3 (comparativa v1=9 / v2b=12 / v2=15 técnicas). Así exp3 puede
recortar subconjuntos (9/12/15) sin depender de los archivos de producción
(pipeline/09_aumentar_datos.py solo genera las 12 técnicas v2b, ×13).

Las técnicas se importan de utils/augmentation.py (TECNICAS_V2, superconjunto
de las 12 de producción v2b más las 3 temporales extremas).

Las 15 técnicas:
  v1  (9) : flip, slow, fast, noise, rot_pos, rot_neg, scale, flip_noise, slow_noise
  v2b (+3): translate, fast_flip, rot_noise
  v2  (+3): very_slow (0.5×), very_fast (1.5×), frame_drop

Prerequisito: 07_extraer_keypoints.py ejecutado (keypoints/label_studio/ poblado).
Salida: experimentos/exp3_tecnicas/salidas/keypoints_aug15/ + dataset_aumentado_15tecnicas.csv
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import DATA_DIR
from utils.augmentation import TECNICAS_V2
from utils.data_io import generar_dataset_aumentado

SALIDA_DIR = os.path.join(DATA_DIR, "experimentos", "exp3_tecnicas", "salidas", "keypoints_aug15")
SALIDA_CSV = os.path.join(DATA_DIR, "experimentos", "exp3_tecnicas", "salidas", "dataset_aumentado_15tecnicas.csv")


def main():
    generar_dataset_aumentado(TECNICAS_V2, SALIDA_DIR, SALIDA_CSV)


if __name__ == "__main__":
    main()
