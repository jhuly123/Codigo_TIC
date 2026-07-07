"""
Función: importa los videos *_cfr.mp4 como tareas en lote en Label Studio, para que anotadores y el
ML Backend puedan trabajar sobre ellas sin carga manual.
"""

import os
import sys
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.label_studio_api import LABEL_STUDIO_URL, API_KEY, PROJECT_ID, HEADERS

PROYECTO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_ROOT   = os.getenv("LSEC_VIDEOS", os.path.join(PROYECTO_ROOT, "videos"))
BATCH_SIZE    = 50


def crear_tareas(video_paths: list) -> list:
    """Formatea lista de rutas como tareas JSON para Label Studio.

    La URL usa ruta RELATIVA desde LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
    (raíz del proyecto). Label Studio concatena internamente:
      DOCUMENT_ROOT + ruta_relativa → archivo físico
    """
    tareas = []
    for ruta in video_paths:
        glosa       = Path(ruta).parent.name
        video_id    = Path(ruta).stem
        # Ruta relativa desde la raíz del proyecto: videos/GLOSA/archivo.mp4
        ruta_rel    = os.path.relpath(ruta, PROYECTO_ROOT).replace(os.sep, '/')
        tareas.append({
            "data": {
                "video":    f"/data/local-files/?d={ruta_rel}",
                "video_id": video_id,
                "glosa":    glosa,
            }
        })
    return tareas


def importar_batch(tareas: list) -> int:
    r = requests.post(
        f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/import",
        headers=HEADERS,
        json=tareas,
    )
    return r.status_code


def main():
    if not API_KEY:
        print("[ERROR] Define LABEL_STUDIO_API_KEY como variable de entorno")
        return

    videos = []
    for glosa in sorted(os.listdir(VIDEOS_ROOT)):
        glosa_path = os.path.join(VIDEOS_ROOT, glosa)
        if not os.path.isdir(glosa_path):
            continue
        for f in os.listdir(glosa_path):
            if f.endswith('_cfr.mp4'):
                videos.append(os.path.join(glosa_path, f))

    print(f"Videos encontrados: {len(videos)}")

    importados = errores = 0
    for i in range(0, len(videos), BATCH_SIZE):
        batch  = videos[i:i + BATCH_SIZE]
        status = importar_batch(crear_tareas(batch))
        if status in (200, 201):
            importados += len(batch)
            print(f"  Batch {i//BATCH_SIZE + 1}: {len(batch)} tareas importadas")
        else:
            errores += len(batch)
            print(f"  [ERROR] Batch {i//BATCH_SIZE + 1}: status {status}")

    print(f"\nImportados: {importados} | Errores: {errores}")


if __name__ == "__main__":
    main()
