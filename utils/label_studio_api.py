# label_studio_api.py — configuración y descarga paginada de tareas,
# compartidas por pipeline/02_import_videos.py, 03_generar_predicciones.py
# y 04_extraer_metadatos.py.
#
# Variables de entorno: LABEL_STUDIO_URL (default localhost:8081),
# LABEL_STUDIO_API_KEY (obligatoria), LABEL_STUDIO_PROJECT (default 5).

import os
import requests

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8081")
API_KEY          = os.getenv("LABEL_STUDIO_API_KEY", "")
PROJECT_ID       = int(os.getenv("LABEL_STUDIO_PROJECT", "5"))

HEADERS = {
    "Authorization": f"Token {API_KEY}",
    "Content-Type": "application/json",
}


def obtener_tareas(project_id: int = PROJECT_ID) -> list:
    """Descarga todas las tareas del proyecto mediante paginación."""
    tareas, page = [], 1
    while True:
        r = requests.get(
            f"{LABEL_STUDIO_URL}/api/tasks"
            f"?project={project_id}&page={page}&page_size=100&fields=all",
            headers=HEADERS,
        )
        batch = r.json().get('tasks', [])
        if not batch:
            break
        tareas.extend(batch)
        print(f"  Página {page}: {len(batch)} tareas — total: {len(tareas)}")
        page += 1
    return tareas
