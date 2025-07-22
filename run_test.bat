@echo off
echo ========================================
echo 🧪 RUNNING CIFAR-10 ACCURACY TEST
echo ========================================
echo.

cd /d D:\new_paper

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Running accuracy validation test...
python simple_accuracy_test.py

echo.
echo ========================================
echo 🏁 TEST COMPLETED
echo ========================================
pause 