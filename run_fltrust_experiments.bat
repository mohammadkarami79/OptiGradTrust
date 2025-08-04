@echo off
chcp 65001 >nul
echo.
echo ===============================================================================
echo                   FLTrust Comprehensive Experiments Runner
echo ===============================================================================
echo.
echo This will run all FLTrust experiments for paper comparison
echo Estimated time: 2-4 hours depending on your hardware
echo.
echo Experiments include:
echo - 3 Datasets: MNIST, CIFAR-10, Alzheimer MRI
echo - 5 Distribution types: IID, Label-Skew (70%%, 90%%), Dirichlet (α=0.5, α=0.1)
echo - 6 Attack scenarios: Clean, Scaling, Partial Scaling, Sign Flipping, 
echo                       Gaussian Noise, Label Flipping
echo - Configuration: 10 clients, 25 rounds, 5 local epochs, 30%% malicious clients
echo.
echo Results will be saved in multiple formats:
echo - JSON: Detailed results with training history
echo - CSV: Summary table for analysis
echo - LaTeX: Ready-to-use table for paper
echo - Log: Detailed execution log
echo.
echo ===============================================================================

choice /C YN /M "Do you want to proceed with all experiments?"
if errorlevel 2 goto :cancel
if errorlevel 1 goto :proceed

:proceed
echo.
echo Starting comprehensive FLTrust experiments...
echo.

:: Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Using system Python...
)

:: Install required packages if needed
echo Installing/checking requirements...
pip install torch torchvision numpy scikit-learn matplotlib --quiet

:: Run the experiments
echo.
echo Starting experiments...
echo.
python run_fltrust_experiments.py

if %errorlevel% equ 0 (
    echo.
    echo ===============================================================================
    echo                        Experiments Completed Successfully!
    echo ===============================================================================
    echo.
    echo Results have been saved in the following files:
    echo - fltrust_results_*.json    ^(Detailed results^)
    echo - fltrust_results_*.csv     ^(Summary table^)
    echo - fltrust_latex_table_*.tex ^(LaTeX table for paper^)
    echo - fltrust_experiments.log   ^(Execution log^)
    echo.
    echo You can now use these results to compare with your paper results.
    echo.
) else (
    echo.
    echo ===============================================================================
    echo                           Experiments Failed!
    echo ===============================================================================
    echo.
    echo Check the log file for error details: fltrust_experiments.log
    echo Partial results may have been saved.
    echo.
)

pause
goto :end

:cancel
echo.
echo Experiments cancelled by user.
echo.
pause

:end 