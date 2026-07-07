# adaptadores.py — cargadores normalizados para las 3 fuentes de etiquetado
# (Label Studio, CVAT, ELAN). Todos devuelven un DataFrame con el mismo
# esquema: herramienta, video_id, glosa, tiempo_inicio, tiempo_fin,
# duracion_seg_etiqueta, confianza, mano_dominante.
#
# Importado por pipeline/05_analizar_dataset.ipynb.

import os
import xml.etree.ElementTree as ET
import pandas as pd

FPS_CVAT = 59.94

CORRECCIONES_GLOSA = {
    # Normalizar al nombre actual usado en Label Studio (fuente de verdad)
    "ENCENDIDO ENCENDIDO": "ENCENDIDO",   # renombrado por el anotador en LS
    "ABAJO":               "ABAJO_2",
    "ESCUCHAR+RUMBLE":     "ESCUCHAR+(1h)RUMBLE",
}


def cargar_label_studio(csv_path: str) -> pd.DataFrame | None:
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df["herramienta"] = "label_studio"
    return df


def _extraer_boxes(track, fps: float, task_start_frame: int = 0) -> dict | None:
    boxes = track.findall("box")
    if not boxes:
        return None
    f_ini = f_fin = None
    for i, box in enumerate(boxes):
        f       = int(box.get("frame")) - task_start_frame   # frame LOCAL del video
        outside = int(box.get("outside", 0))
        if outside == 0 and f_ini is None:
            f_ini = f
        if outside == 1:
            f_fin = (int(boxes[i - 1].get("frame")) - task_start_frame
                     if i > 0 else f_ini)
            break
    if f_fin is None and f_ini is not None:
        f_fin = max(int(b.get("frame")) - task_start_frame for b in boxes
                    if int(b.get("outside", 0)) == 0)
    if f_ini is None or f_fin is None:
        return None
    mano = "derecha"
    attr = boxes[0].find("attribute[@name='mano dominante']")
    if attr is not None and attr.text:
        mano = attr.text.strip()
    t_ini = round(f_ini / fps, 3)
    t_fin = round(f_fin / fps, 3)
    return {"tiempo_inicio": t_ini, "tiempo_fin": t_fin,
            "duracion_seg_etiqueta": round(t_fin - t_ini, 3), "mano": mano}


def _parsear_xml_cvat(xml_path: str, fps: float) -> list:
    """Formato TAREA: un annotations.xml por subcarpeta (video_id = nombre carpeta)."""
    registros = []
    root = ET.parse(xml_path).getroot()
    for track in root.findall("track"):
        label = CORRECCIONES_GLOSA.get(track.get("label"), track.get("label"))
        datos = _extraer_boxes(track, fps)
        if datos:
            registros.append({"label": label, **datos})
    return registros


def _parsear_xml_cvat_proyecto(xml_path: str, fps: float) -> list:
    registros = []
    root = ET.parse(xml_path).getroot()

    # ── Construir mapa task_id → (video_id, global_start_frame) ──────────────
    # Recorrer las tareas EN ORDEN para acumular el offset correcto.
    task_map   = {}   # tid → (video_id, global_start)
    global_start = 0
    for task in root.findall("./meta/project/tasks/task"):
        tid        = task.findtext("id")
        source     = task.findtext("source", "")       # "20250407114253_cfr.mp4"
        stop_frame = int(task.findtext("stop_frame", "0"))
        if tid and source:
            video_id = source.replace(".mp4", "").replace(".MP4", "")
            task_map[tid] = (video_id, global_start)
        # Cada tarea ocupa (stop_frame + 1) slots en la numeración global
        global_start += stop_frame + 1

    for track in root.findall("track"):
        label     = CORRECCIONES_GLOSA.get(track.get("label"), track.get("label"))
        task_id   = track.get("task_id")
        task_info = task_map.get(task_id)
        if task_info is None:
            continue
        video_id, task_start = task_info
        datos = _extraer_boxes(track, fps, task_start_frame=task_start)
        if datos:
            registros.append({"label": label, "video_id": video_id, **datos})
    return registros


def cargar_cvat(carpeta: str, fps: float = FPS_CVAT) -> pd.DataFrame | None:
    if not os.path.isdir(carpeta):
        return None
    filas = []

    # Caso A: XML de proyecto plano — un solo annotations.xml en la raíz de la carpeta
    xml_plano = os.path.join(carpeta, "annotations.xml")
    if os.path.exists(xml_plano):
        for r in _parsear_xml_cvat_proyecto(xml_plano, fps):
            filas.append({
                "herramienta":           "cvat",
                "video_id":              r["video_id"],
                "glosa":                 r["label"],
                "tiempo_inicio":         r["tiempo_inicio"],
                "tiempo_fin":            r["tiempo_fin"],
                "duracion_seg_etiqueta": r["duracion_seg_etiqueta"],
                "confianza":             1.0,
                "mano_dominante":        r["mano"],
            })
        if filas:
            return pd.DataFrame(filas)

    # Caso B: estructura en subdirectorios — un annotations.xml por subcarpeta
    for sub in sorted(os.listdir(carpeta)):
        xml = os.path.join(carpeta, sub, "annotations.xml")
        if not os.path.exists(xml):
            continue
        for t in _parsear_xml_cvat(xml, fps):
            filas.append({
                "herramienta":           "cvat",
                "video_id":              sub,
                "glosa":                 t["label"],
                "tiempo_inicio":         t["tiempo_inicio"],
                "tiempo_fin":            t["tiempo_fin"],
                "duracion_seg_etiqueta": t["duracion_seg_etiqueta"],
                "confianza":             1.0,
                "mano_dominante":        t["mano"],
            })
    return pd.DataFrame(filas) if filas else None


def cargar_elan(carpeta: str) -> pd.DataFrame | None:
    if not os.path.isdir(carpeta):
        return None
    try:
        import pympi
    except ImportError:
        print("  [AVISO] pympi-ling no instalado — pip install pympi-ling")
        return None
    filas = []
    for archivo in sorted(os.listdir(carpeta)):
        if not archivo.endswith(".eaf"):
            continue

        # Extraer video_id del nombre: formato esperado GLOSA__video_id.mp4.eaf
        # o simplemente video_id.eaf (formato legacy).
        fname = archivo.replace(".eaf", "")           # "ABAJO_2__20250407114433_cfr.mp4"
        if "__" in fname:
            video_id = fname.split("__", 1)[1]        # "20250407114433_cfr.mp4"
            video_id = video_id.replace(".mp4", "").replace(".MP4", "")  # "20250407114433_cfr"
        else:
            video_id = fname.replace(".mp4", "").replace(".MP4", "")

        try:
            eaf = pympi.Elan.Eaf(os.path.join(carpeta, archivo))
        except Exception as e:
            print(f"  [ERROR] {archivo}: {e}")
            continue
        if "Glosa" not in eaf.get_tier_names():
            continue
        manos = {}
        if "Mano_Dominante" in eaf.get_tier_names():
            manos = {(a[0], a[1]): a[2]
                     for a in eaf.get_annotation_data_for_tier("Mano_Dominante")}
        for ini_ms, fin_ms, glosa_raw in eaf.get_annotation_data_for_tier("Glosa"):
            glosa = CORRECCIONES_GLOSA.get(glosa_raw.strip(), glosa_raw.strip())
            t_ini = round(ini_ms / 1000.0, 3)
            t_fin = round(fin_ms / 1000.0, 3)
            mano  = "derecha"
            for (m_ini, _), val in manos.items():
                if abs(m_ini - ini_ms) < 100:
                    mano = val.strip()
                    break
            filas.append({
                "herramienta":           "elan",
                "video_id":              video_id,
                "glosa":                 glosa,
                "tiempo_inicio":         t_ini,
                "tiempo_fin":            t_fin,
                "duracion_seg_etiqueta": round(t_fin - t_ini, 3),
                "confianza":             1.0,
                "mano_dominante":        mano,
            })
    return pd.DataFrame(filas) if filas else None
