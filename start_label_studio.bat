@echo off
:: ============================================================
:: Inicia Label Studio con acceso a los videos locales del proyecto
:: Doble clic para ejecutar, o desde Anaconda Prompt
:: ============================================================

:: 1) Activar conda PRIMERO (puede limpiar variables de entorno)
call %USERPROFILE%\miniconda3\Scripts\activate.bat %USERPROFILE%\miniconda3\envs\senas_env

echo.
echo  Iniciando Label Studio con file serving habilitado...
echo  (via start_label_studio.py para garantizar env vars correctas)
echo.

python "%~dp0start_label_studio.py"

pause
