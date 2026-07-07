"""
modelo/evaluar.py
===================
Evalúa el modelo final (modelo/resultados/mejor_modelo_v2.keras) sobre el test set fijo
de Label Studio y sobre las herramientas comparativas (CVAT, ELAN), y genera
los artefactos de modelo/resultados/:

  confusion_matrix.png          — matriz de confusión normalizada (Label Studio)
  confusion_matrix_cvat.png     — matriz de confusión normalizada (CVAT)
  confusion_matrix_elan.png     — matriz de confusión normalizada (ELAN)
  reporte_por_clase.csv         — precision/recall/F1 por seña (test Label Studio)
  comparativa_herramientas.csv  — accuracy y top-5 accuracy por herramienta

Reconstruido en 2026-07-05: el repo ya no tenía un script que regenerara estos
archivos (la guía de ejecución mencionaba modelo/evaluar.py, pero no existía).

No regenera comparativa_herramientas_historico_produccion.csv (salida histórica de
experimentos/exp4_herramientas/exp4_comparativa_herramientas.ipynb): ese
comparaba contra experimentos/exp1_ab/salidas/modelo_B.h5, un artefacto de una
estructura experimental anterior (split único) que el protocolo actual de
5-Fold CV ya no persiste. Reconstruirlo requeriría entrenar y guardar un
modelo adicional, fuera del alcance de este script.

Uso:
    python modelo/evaluar.py
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay,
                              top_k_accuracy_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import (DATA_DIR, N_FRAMES, N_KEYPOINTS, SPLIT_PATH,
                          CSV_TOP35, KEYPOINTS_DIR)
from utils.preprocessing import normalizar_hombros
from utils.data_io import cargar_herramienta_comparativa

SALIDA_DIR  = os.path.join(DATA_DIR, "modelo", "resultados")
CHECKPOINT  = os.path.join(SALIDA_DIR, "mejor_modelo_v2.keras")
os.makedirs(SALIDA_DIR, exist_ok=True)


def cargar_test_label_studio(idx_map: dict) -> tuple:
    """Carga el test set fijo (split_fijo.npz) de Label Studio, normalizado."""
    df_top35 = pd.read_csv(CSV_TOP35)
    X_all, y_all = [], []
    for _, row in df_top35.iterrows():
        npy = os.path.join(KEYPOINTS_DIR, f"{row['video_id']}.npy")
        if not os.path.exists(npy):
            continue
        kp = np.load(npy)
        if kp.shape != (N_FRAMES, N_KEYPOINTS):
            continue
        X_all.append(kp)
        y_all.append(idx_map.get(row['glosa'], -1))
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)

    split    = np.load(SPLIT_PATH, allow_pickle=True)
    idx_test = split['idx_test'].tolist()
    X_test   = np.array([normalizar_hombros(X_all[i]) for i in idx_test], dtype=np.float32)
    y_test   = y_all[idx_test]
    return X_test, y_test


def top5_accuracy(model, X, y, n_clases) -> float | None:
    """Top-5 accuracy; None si hay menos de 5 clases (sklearn lo exige)."""
    if len(X) == 0 or n_clases < 5:
        return None
    prob = model.predict(X, verbose=0)
    return float(top_k_accuracy_score(y, prob, k=5, labels=list(range(n_clases))))


def guardar_matriz_confusion(y_true, y_pred, clases, titulo, nombre_archivo):
    """
    Genera y guarda una matriz de confusión normalizada con el estilo estándar
    del proyecto. Muestra el valor en TODAS las celdas (incluidos ceros); con
    muchas clases (35), el tamaño de figura y de fuente se escalan para que
    el texto no se solape (a 1 decimal alcanza: los valores son 0.0 o 1.0 casi
    siempre, dado que hay ~1 muestra de test por clase).
    """
    n_clases = len(clases)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_clases)), normalize='true')
    fig_size = max(10, n_clases * 0.5)
    fontsize = max(5, min(8, 260 / n_clases))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp.plot(ax=ax, cmap='Blues', values_format='.1f', xticks_rotation=90,
             colorbar=True, text_kw={'fontsize': fontsize})
    ax.set_title(titulo, fontweight='bold')
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    fig.tight_layout()
    ruta = os.path.join(SALIDA_DIR, nombre_archivo)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Guardado: {ruta}")


def main():
    if not os.path.exists(CHECKPOINT):
        print(f"[ERROR] No existe {CHECKPOINT}. Ejecuta primero modelo/entrenar.ipynb")
        return

    print(f"Cargando modelo: {CHECKPOINT}")
    model = tf.keras.models.load_model(CHECKPOINT)

    df_top35 = pd.read_csv(CSV_TOP35)
    clases   = sorted(df_top35['glosa'].unique())
    idx_map  = {c: i for i, c in enumerate(clases)}
    n_clases = len(clases)

    # ─── Test set Label Studio: confusion matrix + reporte por clase ──────────
    X_test, y_test = cargar_test_label_studio(idx_map)
    pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc_test  = accuracy_score(y_test, pred_test)
    print(f"Accuracy test (Label Studio): {acc_test*100:.1f}%  ({len(X_test)} videos)")

    reporte = classification_report(y_test, pred_test, target_names=clases,
                                    output_dict=True, zero_division=0)
    df_reporte = pd.DataFrame(reporte).transpose()
    ruta_reporte = os.path.join(SALIDA_DIR, "reporte_por_clase.csv")
    df_reporte.to_csv(ruta_reporte, encoding="utf-8-sig")
    print(f"Guardado: {ruta_reporte}")

    guardar_matriz_confusion(y_test, pred_test, clases,
                             'Matriz de confusión normalizada — Test Label Studio',
                             'confusion_matrix.png')

    # ─── Comparativa por herramienta (LS / CVAT / ELAN) ────────────────────────
    registros = []
    acc_ls  = acc_test
    top5_ls = top5_accuracy(model, X_test, y_test, n_clases)
    registros.append({'herramienta': 'Label Studio (auto)',
                      'accuracy': acc_ls, 'top5': top5_ls, 'n_videos': len(X_test)})

    for herramienta, etiqueta, archivo_cm in [
        ('cvat', 'CVAT (visual manual)', 'confusion_matrix_cvat.png'),
        ('elan', 'ELAN (lingüístico)',   'confusion_matrix_elan.png'),
    ]:
        X_h, y_h = cargar_herramienta_comparativa(herramienta, idx_map)
        if len(X_h) == 0:
            print(f"  [AVISO] Sin datos para {etiqueta}")
            continue
        pred_h = np.argmax(model.predict(X_h, verbose=0), axis=1)
        acc_h  = accuracy_score(y_h, pred_h)
        top5_h = top5_accuracy(model, X_h, y_h, n_clases)
        print(f"  [{etiqueta}] accuracy={acc_h*100:.1f}%  ({len(X_h)} videos)")
        registros.append({'herramienta': etiqueta, 'accuracy': acc_h,
                          'top5': top5_h, 'n_videos': len(X_h)})
        guardar_matriz_confusion(y_h, pred_h, clases,
                                 f'Matriz de confusión normalizada — {etiqueta}',
                                 archivo_cm)

    df_comp = pd.DataFrame(registros)
    ruta_comp = os.path.join(SALIDA_DIR, "comparativa_herramientas.csv")
    df_comp.to_csv(ruta_comp, index=False, encoding="utf-8-sig")
    print(f"Guardado: {ruta_comp}")

    print("\n=== RESUMEN ===")
    print(df_comp.to_string(index=False))


if __name__ == "__main__":
    main()
