@echo off
setlocal
cd /d "%~dp0"
set "PAPER_PYTHON_BINARY=%~dp0.venv\Scripts\python.exe"
if not exist "%PAPER_PYTHON_BINARY%" (
    echo Missing .venv. Follow the installation commands in README.md first.
    pause
    exit /b 1
)
if "%~1"=="" (
    "%PAPER_PYTHON_BINARY%" run_paper.py input --output-dir outputs\paper
) else (
    "%PAPER_PYTHON_BINARY%" run_paper.py %*
)
set "PAPER_RUN_EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%PAPER_RUN_EXIT_CODE%"=="0" (
    echo Inference did not finish. Check the message above and docs\WEIGHTS.md.
) else (
    echo Done. Open outputs\paper\*_comparison.png to see the three panels.
)
pause
exit /b %PAPER_RUN_EXIT_CODE%
