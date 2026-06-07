@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYTHON_EXE="

if exist "D:\anaconda3\python.exe" (
    set "PYTHON_EXE=D:\anaconda3\python.exe"
)

if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
    )
)

if not defined PYTHON_EXE (
    echo Could not find Python.
    echo Please install Python or Anaconda, then run this file again.
    echo.
    pause
    exit /b 1
)

echo AutoSolver Dashboard Launcher
echo ----------------------------------------
echo Project: %cd%
echo Python : %PYTHON_EXE%
echo.

"%PYTHON_EXE%" -c "import sys; print('Python version:', sys.version.replace(chr(10), ' '))"
if errorlevel 1 (
    echo.
    echo Python could not start.
    pause
    exit /b 1
)

echo.
echo Checking Python packages...
"%PYTHON_EXE%" -c "import importlib.util, sys; missing=[p for p in ('openai','ortools','streamlit','plotly','pandas') if importlib.util.find_spec(p) is None]; print('Missing packages: ' + (', '.join(missing) if missing else 'none')); sys.exit(1 if missing else 0)"
if errorlevel 1 (
    echo.
    echo Installing required packages from requirements.txt...
    echo This may take several minutes the first time.
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed.
        echo Please check the error messages above.
        pause
        exit /b 1
    )
)

echo.
netstat -ano | findstr /R /C:":8501 .*LISTENING" >nul
if not errorlevel 1 (
    echo Port 8501 already has a service listening.
    echo Opening the dashboard URL directly...
    start "" "http://127.0.0.1:8501"
    echo.
    pause
    exit /b 0
)

echo Starting AutoSolver Streamlit dashboard...
echo URL: http://127.0.0.1:8501
echo.
echo Keep this window open while using the dashboard.
echo Press Ctrl+C here to stop it.
echo.

start "" cmd /c "timeout /t 5 /nobreak >nul && start http://127.0.0.1:8501"
"%PYTHON_EXE%" -m streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true

echo.
echo Dashboard stopped. Check the messages above if it did not start.
pause

endlocal
