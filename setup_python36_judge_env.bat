@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo AutoSolver Python 3.6 Judge Env Setup
echo ----------------------------------------
echo This creates .venv36 only for running generated solver.py during evaluation.
echo The dashboard and AutoSolver agent will still run with your normal Python.
echo.

set "PY36_EXE="

if exist ".venv36\Scripts\python.exe" (
    set "PY36_EXE=%cd%\.venv36\Scripts\python.exe"
    goto verify_existing
)

for /f "delims=" %%P in ('py -3.6 -c "import sys; print(sys.executable)" 2^>nul') do (
    if not defined PY36_EXE set "PY36_EXE=%%P"
)

if not defined PY36_EXE (
    for %%P in (
        "C:\Python36\python.exe"
        "C:\Python36-32\python.exe"
        "C:\Program Files\Python36\python.exe"
        "C:\Program Files (x86)\Python36\python.exe"
        "%LocalAppData%\Programs\Python\Python36\python.exe"
        "%LocalAppData%\Programs\Python\Python36-32\python.exe"
    ) do (
        if exist %%~P (
            if not defined PY36_EXE set "PY36_EXE=%%~P"
        )
    )
)

if not defined PY36_EXE (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PY36_EXE (
            "%%P" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 6) else 1)" >nul 2>nul
            if not errorlevel 1 set "PY36_EXE=%%P"
        )
    )
)

if not defined PY36_EXE (
    echo Python 3.6 was not found on this machine.
    echo Please install Python 3.6 x64, then run this file again.
    echo.
    echo After setup, the dashboard will auto-detect:
    echo %cd%\.venv36\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

echo Found Python 3.6:
"%PY36_EXE%" -c "import sys; print(sys.version.replace(chr(10), ' '))"
if errorlevel 1 (
    echo Python 3.6 could not start.
    pause
    exit /b 1
)

echo.
echo Creating .venv36 ...
"%PY36_EXE%" -m venv .venv36
if errorlevel 1 (
    echo Failed to create .venv36.
    pause
    exit /b 1
)

set "PY36_EXE=%cd%\.venv36\Scripts\python.exe"

:verify_existing
echo.
"%PY36_EXE%" -c "import sys; print('Judge Python:', sys.version.replace(chr(10), ' '))"
if errorlevel 1 (
    echo The judge Python could not start.
    pause
    exit /b 1
)

echo.
echo Done.
echo Use this path in the dashboard advanced parameter "solver judge Python":
echo %cd%\.venv36\Scripts\python.exe
echo.
pause

endlocal
