@echo off
echo ========================================
echo WeatherShield AI - Model Training
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python first.
    pause
    exit /b 1
)

REM Check Python version and display message
python --version >tmp_pyver.txt 2>&1
findstr /C:"3.13" tmp_pyver.txt >nul
if %errorlevel%==0 (
    echo ERROR: Python 3.13 is not supported by Streamlit. Please install Python 3.10 or 3.11.
    pause
    exit /b 1
)
del tmp_pyver.txt

REM Check if virtual environment should be created
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install/update required packages
echo Installing required packages...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning
pip install scikit-learn pandas numpy
echo Trying to install torch-geometric...
pip install torch-geometric 2>nul
if errorlevel 1 (
    echo WARNING: torch-geometric installation failed, will use fallback
)

REM Check if data exists
if not exist "data\weather" (
    echo WARNING: No weather data found in data\weather\
    echo Creating dummy data directory...
    mkdir data\weather 2>nul
    echo NOTE: Training will use synthetic data
)

REM Create models directory
mkdir models 2>nul

REM Run training with error handling
echo.
echo Starting model training...
echo Using device: CUDA if available, otherwise CPU
echo.
python training_pipeline.py --stage all --epochs 5 --device cuda
set TRAINING_RESULT=%errorlevel%

REM Check results
echo.
echo ========================================
echo Training Results:
echo ========================================
if %TRAINING_RESULT% == 0 (
    echo ✅ Training completed successfully!
) else (
    echo ⚠️ Training completed with warnings
)

if exist "models\weather_nowcast.ckpt" (
    echo ✅ Weather model: OK
) else (
    echo ❌ Weather model: MISSING
)

if exist "models\grid_gnn.pth" (
    echo ✅ Grid model: OK
) else (
    echo ❌ Grid model: MISSING
)

if exist "models\fusion.pth" (
    echo ✅ Fusion model: OK
) else (
    echo ❌ Fusion model: MISSING
)

if exist "models\scaler.pkl" (
    echo ✅ Scaler: OK
) else (
    echo ❌ Scaler: MISSING
)

echo.
echo You can now run the dashboard: streamlit run dashboard.py
echo.
echo NOTE: If dashboard shows errors with torch._classes, run:
echo pip install streamlit==1.28.1
echo.
pause