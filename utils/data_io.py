# data_io.py — carga de keypoints y gestión del split estratificado fijo.
# El split fijo garantiza que val/test sean idénticos en todos los usos que
# lo comparten, condición necesaria para que las comparativas sean válidas.

import os
import numpy as np
import pandas as pd
from .config import (N_FRAMES, N_KEYPOINTS, CSV_COMP35, CSV_KEYPOINTS,
                     KEYPOINTS_DIR, KP_CVAT_DIR, KP_ELAN_DIR)
from .preprocessing import normalizar_hombros

# Mapa herramienta → directorio de keypoints (generado por 07_extraer_keypoints)
_KP_DIRS = {"cvat": KP_CVAT_DIR, "elan": KP_ELAN_DIR}


def _cargar_npy_de_df(df: pd.DataFrame,
                      kp_dir: str,
                      idx_map: dict,
                      normalizar: bool,
                      etiqueta_dir: str) -> tuple:
    """
    Bucle común: por cada fila con glosa en idx_map carga kp_dir/{video_id}.npy,
    valida el shape (N_FRAMES, N_KEYPOINTS) y apila. Cuenta como omitidos tanto
    los .npy faltantes como los de shape inválido.
    """
    X, y, omitidos = [], [], 0
    for _, row in df.iterrows():
        if row['glosa'] not in idx_map:
            continue
        npy = os.path.join(kp_dir, f"{row['video_id']}.npy")
        if not os.path.exists(npy):
            omitidos += 1
            continue
        kp = np.load(npy)
        if kp.shape != (N_FRAMES, N_KEYPOINTS):
            omitidos += 1
            continue
        X.append(normalizar_hombros(kp) if normalizar else kp)
        y.append(idx_map[row['glosa']])
    if omitidos:
        print(f"  [AVISO] {omitidos} .npy faltantes o con shape inválido en {etiqueta_dir}")
    if not X:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def cargar_keypoints_csv(csv_path: str,
                          kp_dir: str,
                          idx_map: dict,
                          normalizar: bool = True) -> tuple:
    """
    Lee un CSV con columnas `video_id` y `glosa`, carga los .npy desde kp_dir.

    normalizar=True aplica normalizar_hombros — OBLIGATORIO para modelos
    entrenados con datos normalizados; evaluar sin normalizar degrada
    silenciosamente la accuracy.

    Retorna (X float32, y int32). Omite (con aviso) .npy faltantes o con
    shape incorrecto para no interrumpir evaluaciones parciales.
    """
    df = pd.read_csv(csv_path)
    return _cargar_npy_de_df(df, kp_dir, idx_map, normalizar,
                             etiqueta_dir=os.path.basename(kp_dir))


def cargar_herramienta_comparativa(herramienta: str,
                                   idx_map: dict,
                                   normalizar: bool = True) -> tuple:
    """
    Carga keypoints de una herramienta ('cvat' o 'elan') desde
    dataset_comparativa_35.csv, filtrando por herramienta.

    Retorna (X float32, y int32). Arrays vacíos si el CSV no existe o no hay datos.
    """
    kp_dir = _KP_DIRS.get(herramienta)
    if kp_dir is None:
        raise ValueError(f"herramienta desconocida: {herramienta!r}. Usa 'cvat' o 'elan'.")

    if not os.path.exists(CSV_COMP35):
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    df_h = pd.read_csv(CSV_COMP35)
    df_h = df_h[df_h["herramienta"] == herramienta]
    return _cargar_npy_de_df(df_h, kp_dir, idx_map, normalizar,
                             etiqueta_dir=f"keypoints/{herramienta}/")


def generar_dataset_aumentado(tecnicas: dict,
                              salida_dir: str,
                              salida_csv: str,
                              csv_keypoints: str = CSV_KEYPOINTS,
                              kp_dir: str = KEYPOINTS_DIR) -> None:
    """
    Aplica un registro de técnicas de aumento a cada keypoint original y
    persiste el resultado: {video_id}_{tecnica}.npy en salida_dir (más una
    copia {video_id}_original.npy) y un CSV de registro.

    Bucle común de pipeline/09_aumentar_datos.py (TECNICAS_V2B, producción)
    y experimentos/exp3_tecnicas/generar_aumento_15tecnicas.py (TECNICAS_V2,
    dataset aislado del Experimento 3) — la única diferencia entre ambos es
    qué técnicas aplican y dónde escriben.
    """
    os.makedirs(salida_dir, exist_ok=True)
    n_variantes = len(tecnicas) + 1   # + original

    df        = pd.read_csv(csv_keypoints)
    total     = errores = 0
    registros = []

    for _, row in df.iterrows():
        npy_orig = os.path.join(kp_dir, f"{row['video_id']}.npy")
        if not os.path.exists(npy_orig):
            continue
        seq = np.load(npy_orig)
        if seq.shape != (N_FRAMES, N_KEYPOINTS):
            continue

        # Copiar original al nuevo directorio
        ruta_orig = os.path.join(salida_dir, f"{row['video_id']}_original.npy")
        np.save(ruta_orig, seq)
        registros.append({**row.to_dict(), 'tipo': 'original',
                          'original': row['video_id'], 'npy_path': ruta_orig})

        for tipo, fn in tecnicas.items():
            try:
                aug  = fn(seq).astype(np.float32)
                ruta = os.path.join(salida_dir, f"{row['video_id']}_{tipo}.npy")
                np.save(ruta, aug)
                registros.append({**row.to_dict(), 'tipo': tipo,
                                  'original': row['video_id'], 'npy_path': ruta})
                total += 1
            except Exception as e:
                print(f"  [ERROR] {row['video_id']} / {tipo}: {e}")
                errores += 1

        if len(registros) % (n_variantes * 50) == 0:
            print(f"  {len(registros) // n_variantes} videos procesados")

    pd.DataFrame(registros).to_csv(salida_csv, index=False, encoding='utf-8-sig')
    print(f"\nAumentaciones: {total} | Errores: {errores}")
    print(f"Técnicas: {len(tecnicas)} | Factor efectivo: ×{total / max(len(df), 1):.0f} por video original")
    print(f"CSV: {salida_csv}")


def cargar_o_crear_split(split_path: str,
                          y_all: np.ndarray,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15) -> tuple:
    """
    Carga el split estratificado fijo desde disco, o lo crea con seed=42
    si no existe y lo persiste para garantizar reproducibilidad total.

    Usado por pipeline/10_empaquetar_dataset.py y por los notebooks que
    comparten split_fijo.npz. Nota: los scripts 5-fold de experimentos/
    (exp1, exp2, exp4, incremental) usan su propio StratifiedKFold con seed
    fija, no este split.
    """
    if os.path.exists(split_path):
        s = np.load(split_path, allow_pickle=True)
        tr, val, te = (s['idx_train'].tolist(),
                       s['idx_val'].tolist(),
                       s['idx_test'].tolist())
        print(f"Split fijo cargado: train={len(tr)} | val={len(val)} | test={len(te)}")
        return tr, val, te

    np.random.seed(42)
    idx_train, idx_val, idx_test = [], [], []
    for c in np.unique(y_all):
        idx_c = np.where(y_all == c)[0].tolist()
        np.random.shuffle(idx_c)
        n  = len(idx_c)
        nv = max(1, int(n * val_ratio))
        nt = max(1, int(n * test_ratio))
        idx_val.extend(idx_c[:nv])
        idx_test.extend(idx_c[nv:nv + nt])
        idx_train.extend(idx_c[nv + nt:])

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    np.savez(split_path,
             idx_train=np.array(idx_train),
             idx_val=np.array(idx_val),
             idx_test=np.array(idx_test))
    print(f"Split fijo creado: {split_path}")
    print(f"  train={len(idx_train)} | val={len(idx_val)} | test={len(idx_test)}")
    return idx_train, idx_val, idx_test
