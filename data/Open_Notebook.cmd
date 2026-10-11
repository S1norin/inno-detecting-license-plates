@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.13 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m jupyterlab --ServerApp.ip=127.0.0.1 --ServerApp.port=8896 --ServerApp.port_retries=20 --ServerApp.root_dir="%cd%" --ServerApp.default_url=/lab/tree/Data_Preprocessing.ipynb
goto end
:failed
echo Setup failed. Python 3.13 and an internet connection are needed for the first setup.
pause
:end
