"""
Función: convierte videos MTS y MP4 crudos a formato MP4 con framerate constante (CFR).
"""

import os
import json
import subprocess
from pathlib import Path

VIDEOS_ROOT = os.getenv("LSEC_VIDEOS",
                        os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos"))


def obtener_fps(video_path: str) -> tuple:
    """Consulta fps, número de frames y duración via ffprobe."""
    try:
        res = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', video_path],
            capture_output=True, text=True,
            encoding='utf-8', errors='ignore', timeout=10
        )
        for stream in json.loads(res.stdout).get('streams', []):
            if stream.get('codec_type') == 'video':
                num, den = stream.get('r_frame_rate', '30/1').split('/')
                return (round(float(num) / float(den), 2),
                        int(stream.get('nb_frames', 0)),
                        float(stream.get('duration', 0)))
    except Exception as e:
        print(f"  [WARN] ffprobe: {e}")
    return 30.0, 0, 0.0


def convertir_a_cfr(entrada: str, fps: float) -> bool | None:
    """
    Ejecuta ffmpeg para convertir a CFR.
    Retorna True si tuvo éxito, False si falló, None si la salida ya existía.
    """
    base   = Path(entrada).stem.rstrip('-_').strip()
    salida = str(Path(entrada).parent / f"{base}_cfr.mp4")

    if os.path.exists(salida):
        return None  # ya existe, omitir

    resultado = subprocess.run(
        ['ffmpeg', '-i', entrada,
         '-vcodec', 'libx264', '-acodec', 'aac',
         '-r', str(fps), '-vsync', 'cfr',
         '-pix_fmt', 'yuv420p',
         '-profile:v', 'baseline', '-level', '3.1',
         '-movflags', '+faststart',   # moov atom al inicio → reproducible en navegador
         '-y', salida],
        capture_output=True, text=True,
        encoding='utf-8', errors='ignore'
    )
    if resultado.returncode != 0:
        print(f"  [ERROR]\n{resultado.stderr[-300:]}")
        return False
    return True


def main():
    total = errores = omitidos = 0

    for glosa in sorted(os.listdir(VIDEOS_ROOT)):
        glosa_path = os.path.join(VIDEOS_ROOT, glosa)
        if not os.path.isdir(glosa_path):
            continue

        archivos = os.listdir(glosa_path)
        mts = [f for f in archivos if f.upper().endswith('.MTS')]
        mp4 = [f for f in archivos
               if f.lower().endswith('.mp4') and '_cfr' not in f]

        # MTS tiene prioridad; omitir MP4 con mismo timestamp base
        bases_mts = {Path(f).stem.rstrip('-_').strip() for f in mts}
        a_convertir = mts + [
            f for f in mp4
            if Path(f).stem.rstrip('-_').strip() not in bases_mts
        ]

        for archivo in sorted(a_convertir):
            ruta    = os.path.join(glosa_path, archivo)
            fps, nb, dur = obtener_fps(ruta)
            print(f"{glosa}/{archivo}  fps={fps} frames={nb} dur={dur:.1f}s")

            resultado = convertir_a_cfr(ruta, fps)
            if resultado is None:
                print("  [EXISTE]"); omitidos += 1
            elif resultado:
                print("  OK"); total += 1
            else:
                errores += 1

    print(f"\nConvertidos: {total} | Omitidos: {omitidos} | Errores: {errores}")


if __name__ == "__main__":
    main()
