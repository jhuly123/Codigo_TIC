"""
Detecta el segmento activo de una seña en un video mediante velocidad
de muñeca y visibilidad de keypoints MediaPipe Holistic.

Método público:
  detectar_segmento(video_path)
      → ML Backend: encuentra automáticamente inicio y fin de la seña
        en el video completo.

"""

import cv2
import json
import subprocess
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional


@dataclass
class SignSegment:
    """Resultado completo de la detección automática de un segmento."""
    tiempo_inicio:     float
    tiempo_fin:        float
    confianza:         float
    fps:               float
    keypoints_validos: float
    mano_dominante:    str
    velocidad_media:   float
    blur_score:        float
    frame_inicio:      int
    frame_fin:         int
    total_frames:      int
    duracion_total:    float
    resolucion:        str


class SignDetector:
    """
    Detecta y analiza segmentos de lengua de señas en videos.

    Parámetros
    ----------
    umbral_velocidad : velocidad mínima de muñeca para activar un frame
    margen_frames    : frames de margen añadidos al inicio y fin del segmento
    tolerancia_pausa : frames de pausa interna permitidos (pausas naturales)
    """

    def __init__(self, umbral_velocidad=0.005, margen_frames=5, tolerancia_pausa=10):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.umbral     = umbral_velocidad
        self.margen     = margen_frames
        self.tolerancia = tolerancia_pausa

    # ─── Métodos privados ────────────────────────────────────────────────────

    def _metadata_video(self, video_path: str) -> tuple:
        """Retorna (fps, duración, n_frames, resolución) usando ffprobe."""
        try:
            r = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_streams', video_path],
                capture_output=True, text=True,
                encoding='utf-8', errors='ignore', timeout=10,
            )
            for s in json.loads(r.stdout).get('streams', []):
                if s.get('codec_type') == 'video':
                    num, den = s.get('r_frame_rate', '30/1').split('/')
                    fps    = round(float(num) / float(den), 2)
                    dur    = float(s.get('duration', 0))
                    frames = int(s.get('nb_frames', 0))
                    res    = f"{s.get('width', 0)}x{s.get('height', 0)}"
                    return fps, dur, frames, res
        except Exception as e:
            print(f"[WARN] ffprobe: {e}")
        return 30.0, 0.0, 0, "0x0"

    def _analizar_frames(self, cap, f_ini: int, f_fin: int) -> dict:
        """
        Procesa frames en el rango [f_ini, f_fin] del VideoCapture abierto.
        Retorna listas de visibilidad, blur y velocidad de cada muñeca.
        """
        if f_ini > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_ini)

        vis_list  = []
        blur_list = []
        vel_izq   = []
        vel_der   = []
        pos_izq_ant = None
        pos_der_ant = None
        n_max       = (f_fin - f_ini + 1) if f_fin >= f_ini else 999999
        n_leidos    = 0

        while n_leidos < n_max:
            ret, frame = cap.read()
            if not ret:
                break
            n_leidos += 1

            # Nitidez: varianza del Laplaciano (mayor = imagen más nítida)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_list.append(cv2.Laplacian(gray, cv2.CV_64F).var())

            results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            vis = 0.0
            v_i = 0.0
            v_d = 0.0

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Muñeca izquierda = landmark 15
                if lm[15].visibility > 0.5:
                    vis += 0.5
                    pos = np.array([lm[15].x, lm[15].y])
                    if pos_izq_ant is not None:
                        v_i = float(np.linalg.norm(pos - pos_izq_ant))
                    pos_izq_ant = pos
                else:
                    pos_izq_ant = None

                # Muñeca derecha = landmark 16
                if lm[16].visibility > 0.5:
                    vis += 0.5
                    pos = np.array([lm[16].x, lm[16].y])
                    if pos_der_ant is not None:
                        v_d = float(np.linalg.norm(pos - pos_der_ant))
                    pos_der_ant = pos
                else:
                    pos_der_ant = None

            vis_list.append(vis)
            vel_izq.append(v_i)
            vel_der.append(v_d)

        return {'vis': vis_list, 'blur': blur_list,
                'vel_izq': vel_izq, 'vel_der': vel_der}

    def _mano_dominante(self, vel_izq: list, vel_der: list) -> str:
        """Determina la mano más activa por velocidad acumulada total."""
        total_i = sum(vel_izq)
        total_d = sum(vel_der)
        if total_i == 0 and total_d == 0:
            return "ninguna"
        if total_i == 0:
            return "derecha"
        if total_d == 0:
            return "izquierda"
        ratio = total_d / max(total_i, 0.0001)
        if ratio > 1.3:
            return "derecha"
        elif ratio < 0.7:
            return "izquierda"
        else:
            return "ambas_der" if total_d >= total_i else "ambas_izq"

    def _frames_activos(self, vel_izq: list, vel_der: list) -> list:
        """
        Devuelve índices de frames con movimiento superior al umbral.
        Rellena pausas internas menores a tolerancia_pausa frames.
        """
        vel_max   = [max(vel_izq[i], vel_der[i]) for i in range(len(vel_izq))]
        vel_suave = np.convolve(vel_max, np.ones(3) / 3, mode='same').tolist()

        crudos = [i for i, v in enumerate(vel_suave) if v > self.umbral]
        if not crudos:
            return []

        # Rellenar pausas internas
        activos = [crudos[0]]
        for i in range(1, len(crudos)):
            if crudos[i] - crudos[i - 1] <= self.tolerancia:
                activos.extend(range(crudos[i - 1] + 1, crudos[i] + 1))
            else:
                activos.append(crudos[i])
        return activos

    # ─── Métodos públicos ─────────────────────────────────────────────────────

    def detectar_segmento(self, video_path: str) -> Optional[SignSegment]:
        """
        Analiza el video completo y detecta inicio/fin de la seña por velocidad
        de muñeca. Usado por el ML Backend para proponer etiquetas al anotador.
        """
        fps, duracion, total_frames, resolucion = self._metadata_video(video_path)

        cap   = cv2.VideoCapture(video_path)
        datos = self._analizar_frames(cap, 0, total_frames - 1)
        cap.release()

        n = len(datos['vis'])
        if n == 0:
            return None

        activos    = self._frames_activos(datos['vel_izq'], datos['vel_der'])
        kp_validos = sum(datos['vis']) / n

        if not activos:
            # Sin movimiento detectado: usar el 20 %–80 % del video como fallback
            f_ini     = int(n * 0.2)
            f_fin     = int(n * 0.8)
            confianza = 0.2
        else:
            f_ini = max(0, activos[0] - self.margen)
            f_fin = min(n - 1, activos[-1] + self.margen)
            duracion_activa = activos[-1] - activos[0] + 1
            densidad  = min(len(activos) / max(duracion_activa, 1), 1.0)
            confianza = (kp_validos * 0.6) + (densidad * 0.4)

        # Tiempos en segundos
        if duracion > 0 and n > 0:
            t_ini = round((f_ini / n) * duracion, 3)
            t_fin = round((f_fin / n) * duracion, 3)
        else:
            t_ini = round(f_ini / fps, 3)
            t_fin = round(f_fin / fps, 3)

        vel_max = [max(datos['vel_izq'][i], datos['vel_der'][i]) for i in range(n)]

        return SignSegment(
            tiempo_inicio     = t_ini,
            tiempo_fin        = t_fin,
            confianza         = round(float(confianza), 3),
            fps               = fps,
            keypoints_validos = round(kp_validos, 3),
            mano_dominante    = self._mano_dominante(datos['vel_izq'], datos['vel_der']),
            velocidad_media   = round(float(np.mean(vel_max)), 5),
            blur_score        = round(float(np.mean(datos['blur'])), 2),
            frame_inicio      = f_ini,
            frame_fin         = f_fin,
            total_frames      = n,
            duracion_total    = round(duracion if duracion > 0 else n / fps, 3),
            resolucion        = resolucion,
        )

    def __del__(self):
        if hasattr(self, 'holistic'):
            self.holistic.close()
