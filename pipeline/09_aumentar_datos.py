"""
Función: genera 12 variantes de aumento (v2b) por cada keypoint original del dataset.

Las técnicas viven en utils/augmentation.py (TECNICAS_V2B) y son el subconjunto
"v2b" comparado en experimentos/exp3_tecnicas/exp3_tecnicas_temporales.ipynb:
exp3 mostró que v2b generaliza mejor a CVAT/ELAN que v2/15 (las técnicas
temporales extremas —very_slow, very_fast, frame_drop— degradan la robustez
entre herramientas), por eso producción usa v2b. Para reproducir la comparativa
completa v1/v2b/v2 del Experimento 3, ver
experimentos/exp3_tecnicas/generar_aumento_15tecnicas.py.

Técnicas aplicadas (×13 el tamaño del train set, contando el original):
  Geométricas : flip, rot_pos(+10°), rot_neg(-10°), scale, translate
  Temporales  : slow(0.75×), fast(1.25×)
  Estocásticas: noise (ruido gaussiano σ=0.01)
  Combinadas  : flip_noise, slow_noise, fast_flip, rot_noise
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import KP_AUG_DIR, CSV_AUG_V2
from utils.augmentation import TECNICAS_V2B
from utils.data_io import generar_dataset_aumentado


def main():
    generar_dataset_aumentado(TECNICAS_V2B, KP_AUG_DIR, CSV_AUG_V2)


if __name__ == "__main__":
    main()
