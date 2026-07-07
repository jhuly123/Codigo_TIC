"""
Función: envía cada tarea de Label Studio al ML Backend (MediaPipe) y almacena
las predicciones automáticas como sugerencias para el anotador humano.

"""

import os
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.label_studio_api import (LABEL_STUDIO_URL, API_KEY, PROJECT_ID,
                                    HEADERS, obtener_tareas)

ML_BACKEND_URL = os.getenv("ML_BACKEND_URL", "http://localhost:9090")
DELAY_SEG      = 0.05  # pausa entre requests (segundos) para no saturar el backend


def predecir_y_guardar(tarea: dict) -> bool:
    """Llama al ML Backend y almacena la predicción en Label Studio."""
    r = requests.post(
        f"{ML_BACKEND_URL}/predict",
        json={"tasks": [tarea], "project": str(PROJECT_ID)},
    )
    if r.status_code != 200:
        return False

    resultados = r.json().get('results', [])
    if not resultados:
        return False

    pred   = resultados[0]
    result = pred.get('result', [])

    # No guardar predicciones vacías — indica que el backend no detectó la seña
    if not result:
        print(f"    [SKIP] sin segmento detectado para tarea {tarea['id']}")
        return False

    r2 = requests.post(
        f"{LABEL_STUDIO_URL}/api/predictions",
        headers=HEADERS,
        json={
            "task":          tarea['id'],
            "result":        result,
            "score":         pred.get('score', 0.0),
            "model_version": pred.get('model_version', 'mediapipe_holistic_v1'),
            "extra_data":    pred.get('extra_data', {}),
        },
    )
    return r2.status_code in (200, 201)


def main():
    if not API_KEY:
        print("[ERROR] Define LABEL_STUDIO_API_KEY como variable de entorno")
        return

    print("Obteniendo tareas...")
    tareas = obtener_tareas()
    print(f"Total tareas: {len(tareas)}")

    ok = saltadas = errores = 0
    for i, tarea in enumerate(tareas, 1):
        if tarea.get('predictions'):
            saltadas += 1
            continue
        if predecir_y_guardar(tarea):
            ok += 1
        else:
            errores += 1
        time.sleep(DELAY_SEG)
        if i % 10 == 0:
            print(f"  {i}/{len(tareas)} — OK:{ok} Saltadas:{saltadas} Err:{errores}")

    print(f"\nCompletado: {ok} predicciones | {saltadas} ya tenían | {errores} errores")


if __name__ == "__main__":
    main()
