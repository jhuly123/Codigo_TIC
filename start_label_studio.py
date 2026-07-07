"""
Lanza Label Studio con local file serving habilitado.
Ejecutar: python start_label_studio.py  (o start_label_studio.bat)

La raíz de archivos locales es la carpeta de este script (raíz del proyecto),
la misma que debe usar el ML Backend (ver start_ml_backend.bat).
"""
import os
import sys

# Fijar variables en os.environ del mismo proceso Python,
# ANTES de que Django importe los settings.
PROYECTO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ['LOCAL_FILES_SERVING_ENABLED'] = 'true'
os.environ['LOCAL_FILES_DOCUMENT_ROOT'] = PROYECTO_ROOT

print(f" LOCAL_FILES_SERVING_ENABLED = {os.environ['LOCAL_FILES_SERVING_ENABLED']}")
print(f" LOCAL_FILES_DOCUMENT_ROOT   = {os.environ['LOCAL_FILES_DOCUMENT_ROOT']}")
print(f" Iniciando Label Studio en http://localhost:8081 ...")

from label_studio.server import main
sys.argv = ['label-studio', 'start', '--port', '8081']
main()
