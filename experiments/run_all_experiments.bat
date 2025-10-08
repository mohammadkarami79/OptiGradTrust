@echo off
REM ============================================================================
REM OptiGradTrust Experiments Suite - Windows Batch Script
REM ============================================================================
REM
REM Usage:
REM   run_all_experiments.bat all        - Run all experiments
REM   run_all_experiments.bat priority1  - Run critical experiments only
REM   run_all_experiments.bat quick      - Quick test
REM
REM ============================================================================

echo.
echo ============================================================================
echo OptiGradTrust Experiments Suite
echo ============================================================================
echo.

REM Check if virtual environment exists
if exist "%~dp0..\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%~dp0..\venv\Scripts\activate.bat"
) else (
    echo Warning: Virtual environment not found. Using system Python.
)

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

echo.
echo Starting experiments...
echo.

REM Parse command line argument
set MODE=%1
if "%MODE%"=="" set MODE=priority1

if "%MODE%"=="all" (
    echo Running ALL experiments (Priority 1 + 2 + 3)
    python "%~dp0run_all_experiments.py" --all
) else if "%MODE%"=="priority1" (
    echo Running PRIORITY 1 experiments (critical)
    python "%~dp0run_all_experiments.py" --priority 1
) else if "%MODE%"=="priority2" (
    echo Running PRIORITY 1 + 2 experiments
    python "%~dp0run_all_experiments.py" --priority 2
) else if "%MODE%"=="quick" (
    echo Running QUICK TEST
    python "%~dp0run_all_experiments.py" --quick
) else (
    echo Unknown mode: %MODE%
    echo.
    echo Usage:
    echo   run_all_experiments.bat all        - Run all experiments
    echo   run_all_experiments.bat priority1  - Run critical experiments only
    echo   run_all_experiments.bat priority2  - Run priority 1 + 2
    echo   run_all_experiments.bat quick      - Quick test
    exit /b 1
)

echo.
echo ============================================================================
echo Experiments completed!
echo Check experiments/results/ for output files
echo ============================================================================
echo.

pause

