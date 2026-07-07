@echo off
:: ============================================================
:: Inicia el ML Backend (MediaPipe) que genera predicciones automaticas
:: en Label Studio. Debe estar corriendo ANTES de generar predicciones.
:: ============================================================

call %USERPROFILE%\miniconda3\Scripts\activate.bat senas_env

:: La MISMA raiz que usa Label Studio (raiz del proyecto, no videos\):
:: las URLs de las tareas (?d=videos/GLOSA/archivo.mp4) son relativas a ella.
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%~dp0
:: quitar la barra final que agrega %~dp0
if "%LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT:~-1%"=="\" set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT:~0,-1%

cd /d "%~dp0"

echo.
echo  ML Backend iniciando en http://localhost:9090
echo  Raiz de archivos: %LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT%
echo  Presiona Ctrl+C para detener
echo.

python ml_backend\_wsgi.py --port 9090

pause
