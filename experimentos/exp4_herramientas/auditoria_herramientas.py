# -*- coding: utf-8 -*-
"""
auditoria_herramientas.py
=========================
Comparativa rigurosa de herramientas de anotación que corrige los dos
sesgos del análisis básico (exp4_comparativa_herramientas.ipynb):

  1. Evalúa las TRES herramientas sobre EXACTAMENTE los mismos videos
     (intersección de video_ids presentes en las tres fuentes), no sobre
     conjuntos distintos (idx_test para Label Studio vs. 34 comparativos).
  2. Reporta además sobre el subconjunto LIMPIO (videos no vistos en el
     conjunto de entrenamiento bajo el split fijo vigente) y sobre el
     subconjunto estrictamente de test.

Genera las matrices de confusión de CVAT y ELAN sobre el conjunto común.

Modelo evaluado: modelo/resultados/mejor_modelo_v2.keras (producción v2b,
post-fix de flip_seq — el mismo checkpoint que usa modelo/evaluar.py).

Adaptado en 2026-07-08 del script homónimo respaldado en
Respaldo_Limpieza_Proyecto/scripts_fix/ (que evaluaba un modelo pre-fix).

Uso:
    python experimentos/exp4_herramientas/auditoria_herramientas.py

Salida: experimentos/exp4_herramientas/salidas/auditoria/
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.config import (N_FRAMES, N_KEYPOINTS, DATA_DIR, KEYPOINTS_DIR,
                          CSV_TOP35, CSV_COMP35, KP_CVAT_DIR, KP_ELAN_DIR,
                          SPLIT_PATH)
from utils.preprocessing import normalizar_hombros

OUT    = os.path.join(DATA_DIR, "experimentos", "exp4_herramientas",
                      "salidas", "auditoria")
MODELO = os.path.join(DATA_DIR, "modelo", "resultados", "mejor_modelo_v2.keras")
os.makedirs(OUT, exist_ok=True)

KP_DIRS = {"label_studio": KEYPOINTS_DIR, "cvat": KP_CVAT_DIR, "elan": KP_ELAN_DIR}


def top5(y, p):
    return sum(t in np.argsort(pr)[-5:] for t, pr in zip(y, p)) / len(y)


def main():
    if not os.path.exists(MODELO):
        print(f"[ERROR] No existe {MODELO}. Ejecuta primero modelo/entrenar.ipynb")
        return
    print(f"Modelo: {MODELO}")
    model = tf.keras.models.load_model(MODELO)

    df35 = pd.read_csv(CSV_TOP35)
    clases = sorted(df35["glosa"].unique())
    idx_map = {c: i for i, c in enumerate(clases)}
    n_cl = len(clases)

    # --- Membresía (train/val/test) de cada video_id bajo el split vigente ---
    split = np.load(SPLIT_PATH, allow_pickle=True)
    ids_order = []
    for _, row in df35.iterrows():
        p = os.path.join(KEYPOINTS_DIR, f"{row['video_id']}.npy")
        if os.path.exists(p) and np.load(p).shape == (N_FRAMES, N_KEYPOINTS):
            ids_order.append(row["video_id"])
    train_ids = set(ids_order[i] for i in split["idx_train"])
    val_ids   = set(ids_order[i] for i in split["idx_val"])
    test_ids  = set(ids_order[i] for i in split["idx_test"])

    comp = pd.read_csv(CSV_COMP35)
    # video_ids comunes a las 3 herramientas (mismo material de origen)
    ids_por_h = {h: set(comp[comp["herramienta"] == h]["video_id"]) for h in KP_DIRS}
    comunes = sorted(set.intersection(*ids_por_h.values()))
    print(f"Videos comunes a las 3 herramientas: {len(comunes)}")

    glosa_de = dict(zip(comp[comp["herramienta"] == "label_studio"]["video_id"],
                        comp[comp["herramienta"] == "label_studio"]["glosa"]))

    limpios   = [v for v in comunes if v not in train_ids]
    solo_test = [v for v in comunes if v in test_ids]
    print(f"  de los cuales: train={sum(v in train_ids for v in comunes)} "
          f"val={sum(v in val_ids for v in comunes)} test={sum(v in test_ids for v in comunes)}")
    print(f"  conjunto limpio (sin train): {len(limpios)} | solo test: {len(solo_test)}\n")

    def cargar(herramienta, lista):
        kp_dir = KP_DIRS[herramienta]
        X, y = [], []
        for v in lista:
            g = glosa_de.get(v)
            if g not in idx_map:
                continue
            npy = os.path.join(kp_dir, f"{v}.npy")
            if not os.path.exists(npy):
                continue
            kp = np.load(npy)
            if kp.shape != (N_FRAMES, N_KEYPOINTS):
                continue
            X.append(normalizar_hombros(kp)); y.append(idx_map[g])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    filas = []
    for nombre, herr in [("Label Studio", "label_studio"),
                         ("CVAT", "cvat"), ("ELAN", "elan")]:
        for etiqueta, lista in [("mismos videos", comunes),
                                ("limpio (sin train)", limpios),
                                ("solo test", solo_test)]:
            X, y = cargar(herr, lista)
            if len(X) == 0:
                continue
            prob = model.predict(X, verbose=0)
            pred = np.argmax(prob, axis=1)
            acc = accuracy_score(y, pred)
            t5 = top5(y, prob)
            print(f"  {nombre:<13} [{etiqueta:<20}] acc={acc*100:5.1f}%  "
                  f"top5={t5*100:5.1f}%  n={len(X)}")
            filas.append({"herramienta": nombre, "conjunto": etiqueta,
                          "accuracy": round(acc, 4), "top5": round(t5, 4),
                          "n_videos": len(X)})

            if etiqueta == "mismos videos" and herr in ("cvat", "elan"):
                cm = confusion_matrix(y, pred, labels=list(range(n_cl)), normalize="true")
                fig, ax = plt.subplots(figsize=(max(8, n_cl // 2), max(7, n_cl // 2)))
                ConfusionMatrixDisplay(cm, display_labels=clases).plot(
                    ax=ax, xticks_rotation=90, colorbar=True, cmap="Blues",
                    values_format=".1f")
                ax.set_title(f"Matriz de confusión normalizada — {nombre} "
                             f"(modelo v2b, mismos videos)")
                plt.tight_layout()
                ruta = os.path.join(OUT, f"confusion_matrix_{herr}.png")
                plt.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close()
                print(f"      -> matriz guardada: {ruta}")

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(OUT, "comparativa_herramientas_AUDITADA.csv"),
              index=False, encoding="utf-8-sig")

    print("\n=== RESUMEN (mismos videos, evaluación idéntica) ===")
    piv = df[df["conjunto"] == "mismos videos"].set_index("herramienta")["accuracy"] * 100
    for h in ["Label Studio", "CVAT", "ELAN"]:
        if h in piv: print(f"  {h:<13} {piv[h]:.1f}%")
    print("\n=== RESUMEN (solo test, limpio e idéntico) ===")
    pivt = df[df["conjunto"] == "solo test"].set_index("herramienta")["accuracy"] * 100
    for h in ["Label Studio", "CVAT", "ELAN"]:
        if h in pivt:
            n = df[(df.conjunto == "solo test") & (df.herramienta == h)]["n_videos"].iloc[0]
            print(f"  {h:<13} {pivt[h]:.1f}%  (n={n})")

    # Figura de barras: comparativa sobre el conjunto de test limpio e idéntico
    dft = df[df["conjunto"] == "solo test"]
    if len(dft):
        fig, ax = plt.subplots(figsize=(8, 5))
        colores = {"Label Studio": "steelblue", "CVAT": "darkorange", "ELAN": "seagreen"}
        barras = ax.bar(dft["herramienta"], dft["accuracy"] * 100,
                        color=[colores.get(h, "gray") for h in dft["herramienta"]],
                        alpha=0.85)
        for b, v in zip(barras, dft["accuracy"] * 100):
            ax.annotate(f"{v:.1f} %".replace(".", ","),
                        xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")
        n = dft["n_videos"].iloc[0]
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Comparativa sobre conjunto de prueba limpio e idéntico "
                     f"(n={n} videos, modelo v2b)", fontweight="bold")
        ax.set_ylim(0, max(dft["accuracy"] * 100) * 1.3)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        ruta_fig = os.path.join(OUT, "comparativa_test_limpio.png")
        plt.savefig(ruta_fig, dpi=150, bbox_inches="tight"); plt.close()
        print(f"\nFigura de barras guardada: {ruta_fig}")

    print(f"Archivos en: {OUT}")


if __name__ == "__main__":
    main()
