# -*- coding: utf-8 -*-
"""
generar_figura_sobreajuste.py
=============================
Figura de la tesis (Figura 3.3): efecto del aumento de datos sobre el
sobreajuste. Dos paneles:

  izquierda — curvas train vs val del modelo SIN aumento (historial de
              modelo/entrenar_sin_aumento.py), con la brecha sombreada.
  derecha   — comparativa de val_accuracy: sin aumento vs modelo de
              producción v2b (historial de modelo/entrenar.ipynb).

Adaptado en 2026-07-08 del script _gen_overfitting_fig.py respaldado en
Respaldo_Limpieza_Proyecto/ (que usaba historiales pre-fix de flip_seq y
un modelo de 9 técnicas ya descartado).

Prerequisitos: modelo/entrenar_sin_aumento.py y modelo/entrenar.ipynb
ejecutados (historial_sin_aumento.npy e historial_v2.npy presentes).

Uso:
    python modelo/generar_figura_sobreajuste.py

Salida: modelo/resultados/curvas_sobreajuste_v2b.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import DATA_DIR

RESULTADOS = os.path.join(DATA_DIR, "modelo", "resultados")

h_sin = np.load(os.path.join(RESULTADOS, "historial_sin_aumento.npy"),
                allow_pickle=True).item()
h_v2b = np.load(os.path.join(RESULTADOS, "historial_v2.npy"),
                allow_pickle=True).item()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Efecto del aumento de datos sobre el sobreajuste",
             fontweight="bold", fontsize=13)

# ── Panel izquierdo: curvas train vs val SIN aumento ─────────────────────────
ax = axes[0]
ep    = range(1, len(h_sin["loss"]) + 1)
tr    = [v * 100 for v in h_sin["accuracy"]]
va    = [v * 100 for v in h_sin["val_accuracy"]]
ax.plot(ep, tr, color="#d62728", linewidth=1.8, label="Entrenamiento")
ax.plot(ep, va, color="#d62728", linewidth=1.8, linestyle="--", label="Validación")
ax.fill_between(ep, va, tr,
                where=[t > v for t, v in zip(tr, va)],
                alpha=0.15, color="#d62728", label="Brecha (sobreajuste)")
ax.axhline(max(va), color="#d62728", linewidth=0.7, linestyle=":")
ax.set_title("Sin aumento de datos (83 secuencias)", fontsize=11)
ax.set_xlabel("Época"); ax.set_ylabel("Exactitud (%)"); ax.set_ylim(0, 100)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.annotate(
    f"Train máx: {max(tr):.1f} %\nVal máx:   {max(va):.1f} %\nBrecha:    {max(tr) - max(va):.1f} pp",
    xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

# ── Panel derecho: val_accuracy — sin aumento vs producción v2b ──────────────
ax = axes[1]
configs = [
    (h_sin, "Sin aumento (83 secuencias) — val",          "#d62728", "--", 1.5),
    (h_v2b, "Producción v2b (12 técnicas, ×13) — val",    "#1f77b4", "-",  2.0),
]
for h, etiq, col, ls, lw in configs:
    va_c = [v * 100 for v in h["val_accuracy"]]
    ax.plot(range(1, len(va_c) + 1), va_c, color=col, linestyle=ls,
            linewidth=lw, label=etiq)
    best = max(va_c)
    ax.annotate(f"{best:.1f} %", xy=(len(va_c), va_c[-1]),
                xytext=(3, 0), textcoords="offset points",
                fontsize=8, color=col, va="center")

ax.set_title("Comparativa val_accuracy: sin aumento vs con aumento", fontsize=11)
ax.set_xlabel("Época"); ax.set_ylabel("Exactitud de validación (%)")
ax.set_ylim(0, 70); ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(RESULTADOS, "curvas_sobreajuste_v2b.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"OK: {out}")
print(f"  Sin aumento : train máx {max(tr):.1f} % | val máx {max(va):.1f} % | "
      f"brecha {max(tr) - max(va):.1f} pp")
va2 = [v * 100 for v in h_v2b["val_accuracy"]]
tr2 = [v * 100 for v in h_v2b["accuracy"]]
print(f"  Producción  : train máx {max(tr2):.1f} % | val máx {max(va2):.1f} % | "
      f"brecha {max(tr2) - max(va2):.1f} pp")
print(f"  Mejora en val máx: +{max(va2) - max(va):.1f} pp")
