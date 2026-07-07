# Pipeline de Etiquetado, Preprocesamiento y Aumento de Datos para LSEc

**ANEXO I Codigo de pipeline de Trabajo de Integración Curricular**  
Jhuliet Abigail Conza Chancosa | Facultad de Ingeniería Eléctrica y Electrónica

---

## Descripción

Pipeline completo para estandarizar y aumentar un corpus de videos de **Lengua de Señas Ecuatoriana (LSEc)**, generando un dataset de secuencias de keypoints listo para entrenar modelos de clasificación. 

**Resultado principal:** el pipeline mejora la accuracy media de 40,4 % (datos crudos) a 50,3 % (pipeline completo, +9,9 pp) en validación cruzada 5-Fold sobre 35 señas.

---

## Estructura del repositorio

```
Codigo_TIC/
├── pipeline/               # Scripts numerados del pipeline (01 → 10)
├── utils/                  # Módulos reutilizables (config, augmentation, model...)
├── ml_backend/             # Backend MediaPipe para Label Studio
├── modelo/                 # Entrenamiento y evaluación del clasificador LSTM
│   └── resultados/         # Métricas y figuras finales del modelo
├── demo/                   # Demo en tiempo real con webcam
├── experimentos/           # Un subdirectorio por experimento (notebook + resultados)
│   ├── exp1_ab/            # A/B: datos crudos vs pipeline (5-Fold CV)
│   ├── exp2_factorial/     # Diseño factorial 2×2 normalización × aumento
│   ├── exp3_tecnicas/      # Comparativa de técnicas de aumento (v1/v2b/v2)
│   ├── exp4_herramientas/  # Comparativa Label Studio / CVAT / ELAN
│   └── exp_ablacion/       # Análisis incremental del pipeline (C0 → C2 → C3)
├── anotaciones/            # Archivos de etiquetado (Label Studio, CVAT, ELAN)
└── datos/                  # CSVs intermedios del pipeline (no versionados)
```

---

## Requisitos

```bash
conda create -n senas_env python=3.10.13 -y
conda activate senas_env
pip install -r requirements.txt
```

Componentes adicionales: **FFmpeg 6.x**, **Label Studio 1.12.0**, **Docker** (para CVAT local).

---

## Ver resultados sin ejecutar

Los notebooks ya contienen las salidas guardadas. Puedes explorarlos directamente en GitHub:

| Experimento | Notebook | Resultado clave |
|---|---|---|
| A/B: datos crudos vs pipeline | `experimentos/exp1_ab/exp1_ab_5fold.ipynb` | 40,4 % → 50,3 % (+9,9 pp, B gana en 5/5 folds) |
| Diseño factorial 2×2 | `experimentos/exp2_factorial/exp2_factorial_5fold.ipynb` | Sinergia normalización × aumento (+7,8 pp de interacción) |
| Comparativa técnicas de aumento | `experimentos/exp3_tecnicas/exp3_tecnicas_temporales.ipynb` | v2b (12 técnicas) óptima — producción actual |
| Análisis incremental del pipeline | `experimentos/exp_ablacion/exp_ablacion_pipeline.ipynb` | C0→C3: 40,4 % → 50,3 % |
| Comparativa herramientas | `experimentos/exp4_herramientas/exp4_comparativa_herramientas.ipynb` | Label Studio ≈ CVAT > ELAN |
| Modelo final | `modelo/modelo_final.ipynb` | CV robusta 50,3 % [42,5–58,1 %], test 48,6 %, 88 195 parámetros |

> **Nota de reproducibilidad:** todos los notebooks de entrenamiento fuerzan determinismo real en CPU (`TF_ENABLE_ONEDNN_OPTS=0` + hilos intra/inter-op = 1, SEED=42), por lo que re-ejecutarlos reproduce estos números exactamente. El notebook `exp1_ab_pipeline.ipynb` conserva una metodología anterior (split único) y se incluye como registro histórico con sus salidas guardadas.

---

## Datos

Los **videos originales** pertenecen al proyecto de investigación **PIEX-CEDIA-24-27** y no se distribuyen públicamente.

Los **keypoints extraídos** y el **dataset empaquetado** (necesarios para reproducir los experimentos sin reejecutar el pipeline completo) están disponibles en:

> 📦 [Descargar datos — OneDrive](#) *(solicitar acceso a jhuliet.conza@epn.edu.ec)*

### Reproducir desde cero (con videos)
Seguir los scripts `pipeline/01` → `pipeline/10` en orden. 

### Reproducir solo los experimentos (sin videos)
Descargar el paquete de datos del link anterior (carpetas `datos/`, `keypoints/`, `keypoints_aumentado_v2/` y `dataset_final_v2/`) y ejecutar directamente los notebooks en `experimentos/`.

### Demo en tiempo real
Con el modelo entrenado (`modelo/mejor_modelo_v2.keras`) y una webcam:

```bash
python demo/demo_webcam.py
```

Arranca en modo guiado (muestra la seña objetivo y marca en verde cuando la
predicción coincide; navega entre señas con `A`/`D`). Con `M` cambia a modo
libre (reconocimiento abierto sobre las 35 señas).

---

## Arquitectura del modelo

```
Input (60 frames × 225 keypoints)
  └── LSTM(64, L2=0.005) → BatchNorm → Dropout(0.5)
  └── LSTM(32, L2=0.005) → BatchNorm → Dropout(0.5)
  └── Dense(35, softmax)
```

Keypoints: MediaPipe Holistic — pose (99) + mano izq. (63) + mano der. (63) = 225 valores/frame.

---

## Herramientas de etiquetado comparadas

| Herramienta | Videos etiquetados | Tiempo por video | Accuracy modelo final |
|---|---|---|---|
| Label Studio | 1 053 (automático) | ~6 h total | 48,6 % (top-5: 77,1 %) |
| CVAT | 51 (manual) | ~2 min/video | 35,3 % (top-5: 52,9 %) |
| ELAN | 35 (manual) | ~3 min/video | 29,4 % (top-5: 44,1 %) |

---

