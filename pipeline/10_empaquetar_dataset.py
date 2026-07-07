"""
Función: construye los arrays train/val/test a partir del split fijo y los
keypoints originales + aumentados v2.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import (N_FRAMES, N_KEYPOINTS, DATA_DIR, SPLIT_PATH,
                           CSV_TOP35, CSV_AUG_V2, KEYPOINTS_DIR,
                           KP_AUG_DIR, DATASET_V2_DIR)
from utils.data_io import cargar_o_crear_split

os.makedirs(DATASET_V2_DIR, exist_ok=True)


def cargar_kp(df: pd.DataFrame) -> tuple:
    """Carga .npy desde df['npy_path']. Retorna (X float32, y int32)."""
    X, y, faltantes = [], [], 0
    for _, row in df.iterrows():
        npy = row['npy_path']
        if not os.path.exists(npy):
            faltantes += 1; continue
        kp = np.load(npy)
        if kp.shape != (N_FRAMES, N_KEYPOINTS): continue
        X.append(kp); y.append(int(row['clase_idx']))
    if faltantes:
        print(f"  [AVISO] {faltantes} archivos no encontrados")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    # Solo CSV_AUG_V2 es obligatorio previo; SPLIT_PATH se crea automáticamente
    # por cargar_o_crear_split si no existe (seed=42, estratificado).
    for req in [CSV_AUG_V2]:
        if not os.path.exists(req):
            print(f"[ERROR] No existe: {req}"); return
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)

    df_top35  = pd.read_csv(CSV_TOP35)
    clases    = sorted(df_top35['glosa'].unique())
    idx_map   = {c: i for i, c in enumerate(clases)}
    print(f"Clases: {len(clases)}")

    X_all, y_all, video_ids = [], [], []
    for _, row in df_top35.iterrows():
        npy = os.path.join(KEYPOINTS_DIR, f"{row['video_id']}.npy")
        if not os.path.exists(npy): continue
        kp = np.load(npy)
        if kp.shape != (N_FRAMES, N_KEYPOINTS): continue
        X_all.append(kp); y_all.append(idx_map[row['glosa']])
        video_ids.append(row['video_id'])
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)
    print(f"Videos cargados: {len(X_all)}")

    idx_train, idx_val, idx_test = cargar_o_crear_split(SPLIT_PATH, y_all)
    video_ids_train = {video_ids[i] for i in idx_train}

    df_aug = pd.read_csv(CSV_AUG_V2)
    df_aug = df_aug[
        (df_aug['tipo'] != 'original') &
        (df_aug['original'].isin(video_ids_train)) &
        (df_aug['glosa'].isin(clases))
    ].copy()
    df_aug['clase_idx'] = df_aug['glosa'].map(idx_map)
    print(f"Técnicas: {sorted(df_aug['tipo'].unique())}")

    X_aug, y_aug = cargar_kp(df_aug)
    X_train = np.concatenate([X_all[idx_train], X_aug])
    y_train = np.concatenate([y_all[idx_train], y_aug])

    print(f"\n=== SPLIT FINAL ===")
    print(f"Train: {X_train.shape}  (×{len(X_train)//len(idx_train)} factor aumento)")
    print(f"Val:   {X_all[idx_val].shape}")
    print(f"Test:  {X_all[idx_test].shape}")

    for name, arr in [('X_train', X_train), ('y_train', y_train),
                       ('X_val',   X_all[idx_val]),  ('y_val',   y_all[idx_val]),
                       ('X_test',  X_all[idx_test]), ('y_test',  y_all[idx_test])]:
        np.save(os.path.join(DATASET_V2_DIR, f'{name}.npy'), arr)
    np.save(os.path.join(DATASET_V2_DIR, 'clases.npy'), np.array(clases))
    print(f"\nDataset guardado en: {DATASET_V2_DIR}")


if __name__ == "__main__":
    main()
