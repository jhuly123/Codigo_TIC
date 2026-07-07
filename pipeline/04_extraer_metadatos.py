"""
Función: extrae etiquetas y métricas de calidad de todas las tareas de
Label Studio; por cada tarea obtiene la anotación humana (con prioridad
sobre la predicción del ML Backend) junto con FPS, blur score, keypoints
válidos, mano dominante y velocidad media, calculados con
utils/metricas_video.analizar_segmento para el segmento etiquetado.

"""

import os
import sys
import mediapipe as mp
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metricas_video import resolver_video_path, analizar_segmento
from utils.label_studio_api import API_KEY, obtener_tareas

DATA_DIR    = os.getenv("LSEC_DATA_DIR", str(Path(__file__).parent.parent))
VIDEOS_ROOT = os.path.join(DATA_DIR, "videos")
SALIDA_CSV  = os.path.join(DATA_DIR, "anotaciones", "label_studio", "dataset_etiquetas_metadatos.csv")


def extraer_anotacion(tarea: dict) -> dict | None:
    """
    Extrae etiqueta, tiempo inicio y fin de una tarea.
    Prioriza la anotación humana sobre la predicción automática.
    """
    if tarea.get('annotations'):
        resultado = tarea['annotations'][0].get('result', [])
        fuente    = 'human'
    elif tarea.get('predictions'):
        resultado = tarea['predictions'][0].get('result', [])
        fuente    = 'auto'
    else:
        return None

    if not resultado:
        return None

    data     = tarea.get('data', {})
    video_id = data.get('video_id', '')
    glosa    = data.get('glosa', '')

    t_ini         = None
    t_fin         = None
    glosa_anotada = glosa

    for item in resultado:
        value = item.get('value', {})
        tipo  = item.get('type', '')
        campo = item.get('from_name', '')

        if tipo == 'choices':
            glosa_anotada = value.get('choices', [glosa])[0]
        elif campo in ('tiempo_inicio', 'inicio'):
            t_ini = value.get('number')
        elif campo in ('tiempo_fin', 'fin'):
            t_fin = value.get('number')

    # Para predicciones automáticas, buscar tiempos en extra_data si no están en result
    confianza = 0.5
    if fuente == 'auto':
        pred      = tarea['predictions'][0]
        extra     = pred.get('extra_data', {})
        confianza = pred.get('score', 0.5)
        if t_ini is None:
            t_ini = extra.get('tiempo_inicio')
        if t_fin is None:
            t_fin = extra.get('tiempo_fin')

    if t_ini is None or t_fin is None:
        return None

    video_path = resolver_video_path(video_id, glosa_anotada, VIDEOS_ROOT)

    return {
        'video_id':              video_id,
        'glosa':                 glosa_anotada,
        'tiempo_inicio':         t_ini,
        'tiempo_fin':            t_fin,
        'duracion_seg_etiqueta': round(t_fin - t_ini, 3),
        'confianza':             confianza,
        'fuente':                fuente,
        'video_path':            video_path or '',
        'archivo_existe':        bool(video_path),
        'fps':                   None,
        'keypoints_validos':     None,
        'mano_dominante':        None,
        'blur_score':            None,
        'velocidad_media':       None,
    }


def main():
    if not API_KEY:
        print("[ERROR] Define LABEL_STUDIO_API_KEY como variable de entorno")
        return

    print("Descargando tareas de Label Studio...")
    tareas = obtener_tareas()
    print(f"Total: {len(tareas)} tareas")

    registros = []
    errores   = 0
    ok_metr   = 0

    # Inicializar una sola vez: recargar Holistic en cada iteración sería prohibitivamente lento.
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        for i, tarea in enumerate(tareas):
            try:
                reg = extraer_anotacion(tarea)
                if reg is None:
                    continue

                if reg['archivo_existe']:
                    metricas = analizar_segmento(
                        reg['video_path'],
                        reg['tiempo_inicio'],
                        reg['tiempo_fin'],
                        holistic=holistic,   # reutiliza la instancia
                    )
                    if metricas:
                        reg['fps']               = metricas['fps']
                        reg['keypoints_validos'] = metricas['keypoints_validos']
                        reg['mano_dominante']    = metricas['mano_dominante']
                        reg['blur_score']        = metricas['blur_score']
                        reg['velocidad_media']   = metricas['velocidad_media']
                        ok_metr += 1

                registros.append(reg)

                # Guardado parcial cada 50 videos para recuperación ante fallos
                if len(registros) % 50 == 0:
                    pd.DataFrame(registros).to_csv(
                        SALIDA_CSV + '.parcial', index=False, encoding='utf-8-sig')
                    print(f"  {len(registros)} procesados ({ok_metr} con metricas)")

            except Exception as e:
                print(f"  [ERROR] tarea {tarea.get('id')}: {e}")
                errores += 1
    finally:
        holistic.close()   # siempre liberar MediaPipe aunque falle

    df = pd.DataFrame(registros)
    df.to_csv(SALIDA_CSV, index=False, encoding='utf-8-sig')

    if os.path.exists(SALIDA_CSV + '.parcial'):
        os.remove(SALIDA_CSV + '.parcial')

    print(f"\n=== COMPLETADO ===")
    print(f"Registros:  {len(df)}")
    if len(df) > 0:
        print(f"Senias:     {df['glosa'].nunique()}")
        n_metr = df['keypoints_validos'].notna().sum()
        n_nopath = (~df['archivo_existe'].fillna(False)).sum()
        print(f"Con metricas calculadas: {n_metr}/{len(df)}")
        if n_nopath > 0:
            print(f"Videos no encontrados:   {n_nopath} (metricas no calculadas)")
    print(f"Errores:    {errores}")
    print(f"Guardado:   {SALIDA_CSV}")
    print(f"\nSiguiente paso: notebook pipeline/05_analizar_dataset.ipynb")


if __name__ == "__main__":
    main()
