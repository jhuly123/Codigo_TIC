"""
Backend de ML para Label Studio: propone inicio/fin de la seña
automáticamente usando SignDetector (análisis de velocidad de muñeca).

Al abrir una tarea, el anotador recibe una propuesta de tiempo_inicio
y tiempo_fin que solo necesita revisar y confirmar.

Inicio (forma canónica): start_ml_backend.bat
  (equivale a: python ml_backend\\_wsgi.py --port 9090)

"""

import os
from dataclasses import asdict
from label_studio_ml.model import LabelStudioMLBase
from sign_detector import SignDetector


class SignLanguageBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector     = SignDetector(
            umbral_velocidad=0.005,
            margen_frames=5,
            tolerancia_pausa=10,
        )
        # Debe coincidir con LOCAL_FILES_DOCUMENT_ROOT de Label Studio (raíz
        # del proyecto): las URLs ?d=videos/... son relativas a esa raíz.
        self.dataset_root = os.getenv(
            'LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT',
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

    def _resolver_ruta(self, data: dict) -> str:
        """Convierte la URL de Label Studio en ruta local del archivo."""
        video_url = data.get('video', '')
        if '?d=' in video_url:
            ruta_rel = video_url.split('?d=')[1]
            ruta_rel = ruta_rel.replace('%20', ' ').replace('%2B', '+')
            ruta_rel = ruta_rel.replace('/', os.sep)
            return os.path.join(self.dataset_root, ruta_rel)
        return video_url

    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            video_path = self._resolver_ruta(task['data'])
            glosa      = os.path.basename(os.path.dirname(video_path))

            print(f"[INFO] {glosa} -> {os.path.basename(video_path)}")

            if not os.path.exists(video_path):
                print(f"[WARN] Archivo no encontrado: {video_path}")
                predictions.append({'result': [], 'score': 0.0,
                                    'model_version': 'mediapipe_holistic_v1'})
                continue

            try:
                seg = self.detector.detectar_segmento(video_path)
            except Exception as e:
                print(f"[ERROR] {e}")
                predictions.append({'result': [], 'score': 0.0,
                                    'model_version': 'mediapipe_holistic_v1'})
                continue

            # Resultado base: la glosa de la carpeta del video
            result = [{
                'type':      'choices',
                'value':     {'choices': [glosa]},
                'from_name': 'glosa',
                'to_name':   'video',
            }]

            if seg:
                result.append({
                    'type':      'number',
                    'value':     {'number': seg.tiempo_inicio},
                    'from_name': 'tiempo_inicio',
                    'to_name':   'video',
                })
                result.append({
                    'type':      'number',
                    'value':     {'number': seg.tiempo_fin},
                    'from_name': 'tiempo_fin',
                    'to_name':   'video',
                })
                print(f"[INFO] {seg.tiempo_inicio}s -> {seg.tiempo_fin}s  "
                      f"conf={seg.confianza}  mano={seg.mano_dominante}")

            predictions.append({
                'result':        result,
                'score':         seg.confianza if seg else 0.0,
                'model_version': 'mediapipe_holistic_v1',
                'extra_data':    asdict(seg) if seg else {},
            })

        return predictions
