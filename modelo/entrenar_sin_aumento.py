# -*- coding: utf-8 -*-
"""
entrenar_sin_aumento.py
=======================
Entrena el MISMO modelo LSTM de producción (misma arquitectura, semilla,
split fijo y protocolo determinista que modelo/entrenar.ipynb) pero usando
únicamente las 83 secuencias originales de entrenamiento, SIN aumento de
datos. Su único propósito es generar la curva de referencia "antes del
aumento" para la figura de sobreajuste (Figura 3.3 de la tesis).

No toca ningún artefacto de producción: guarda su historial y métricas en
archivos propios (historial_sin_aumento.npy) dentro de modelo/resultados/.

Uso:
    python modelo/entrenar_sin_aumento.py
"""
import os

SEED = 42
os.environ['PYTHONHASHSEED']         = str(SEED)
os.environ['TF_DETERMINISTIC_OPS']   = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['CUDA_VISIBLE_DEVICES']   = ''   # fuerza CPU para maximo determinismo
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'  # evita reduccion multi-hilo no determinista

import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import (DATA_DIR, N_FRAMES, N_KEYPOINTS, SPLIT_PATH,
                          CSV_TOP35, KEYPOINTS_DIR)
from utils.model import construir_lstm, callbacks_entrenamiento

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SALIDA_DIR = os.path.join(DATA_DIR, 'modelo', 'resultados')
os.makedirs(SALIDA_DIR, exist_ok=True)


def main():
    # Carga de las secuencias originales (ya normalizadas por hombros en la
    # extracción, pipeline/07) y del split fijo — idéntico a 10_empaquetar.
    df_top35 = pd.read_csv(CSV_TOP35)
    clases   = sorted(df_top35['glosa'].unique())
    idx_map  = {c: i for i, c in enumerate(clases)}
    n_clases = len(clases)

    X_all, y_all = [], []
    for _, row in df_top35.iterrows():
        npy = os.path.join(KEYPOINTS_DIR, f"{row['video_id']}.npy")
        if not os.path.exists(npy):
            continue
        kp = np.load(npy)
        if kp.shape != (N_FRAMES, N_KEYPOINTS):
            continue
        X_all.append(kp); y_all.append(idx_map[row['glosa']])
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)

    split     = np.load(SPLIT_PATH, allow_pickle=True)
    idx_train = split['idx_train'].tolist()
    idx_val   = split['idx_val'].tolist()
    idx_test  = split['idx_test'].tolist()

    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val,   y_val   = X_all[idx_val],   y_all[idx_val]
    X_test,  y_test  = X_all[idx_test],  y_all[idx_test]
    print(f'Train : {X_train.shape}  (sin aumento)')
    print(f'Val   : {X_val.shape}')
    print(f'Test  : {X_test.shape}')
    print(f'Clases: {n_clases}')

    # Mismo modelo e hiperparámetros que modelo/entrenar.ipynb
    model = construir_lstm(n_clases=n_clases, lr=0.001, dropout=0.5,
                           regularizacion=0.005, capa_densa=False)

    history = model.fit(
        X_train, to_categorical(y_train, n_clases),
        validation_data=(X_val, to_categorical(y_val, n_clases)),
        epochs=100,
        batch_size=32,
        callbacks=callbacks_entrenamiento(patience_stop=20, patience_lr=7),
        verbose=2,
    )

    h  = history.history
    tr = [v * 100 for v in h['accuracy']]
    va = [v * 100 for v in h['val_accuracy']]
    pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc_test  = accuracy_score(y_test, pred_test) * 100

    np.save(os.path.join(SALIDA_DIR, 'historial_sin_aumento.npy'), h)

    print('\n=== RESUMEN (modelo SIN aumento de datos) ===')
    print(f'  Épocas entrenadas       : {len(tr)}')
    print(f'  Train accuracy máx      : {max(tr):.1f} %')
    print(f'  Val accuracy máx        : {max(va):.1f} %  (época {va.index(max(va)) + 1})')
    print(f'  Brecha train-val (máx)  : {max(tr) - max(va):.1f} pp')
    print(f'  Test accuracy           : {acc_test:.1f} %')
    print(f"\nHistorial guardado en: {os.path.join(SALIDA_DIR, 'historial_sin_aumento.npy')}")


if __name__ == '__main__':
    main()
