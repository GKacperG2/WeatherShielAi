@echo off
cls
echo ========================================
echo    WeatherShield AI Control Panel
echo ========================================
echo.
echo Select an option:
echo.
echo 1. Run Dashboard Only (Recommended)
echo 2. Run Full System (API + Dashboard)
echo 3. Download Training Data
echo 4. Train AI Models
echo 5. Install Dependencies
echo 6. Fix Streamlit Issues
echo 7. Debug Dashboard
echo 8. Exit
echo.

set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto dashboard_only
if "%choice%"=="2" goto full_system
if "%choice%"=="3" goto download_data
if "%choice%"=="4" goto train_models
if "%choice%"=="5" goto install_deps
if "%choice%"=="6" goto fix_streamlit
if "%choice%"=="7" goto debug_dashboard
if "%choice%"=="8" goto end

echo Invalid choice!
pause
goto :eof

:dashboard_only
cls
echo Starting Dashboard...
python -m streamlit run dashboard.py
goto end

:full_system
cls
echo Starting Full System...
echo.
echo Starting API server in new window...
start cmd /k "python main.py"
timeout /t 3 >nul
echo Starting Dashboard...
python -m streamlit run dashboard.py
goto end

:download_data
cls
call download_data.bat
goto end

:train_models
cls
call train_models.bat
goto end

:install_deps
cls
echo Installing all dependencies...
python -m pip install -r requirements.txt
echo.
echo Installation complete!
pause
goto end

:fix_streamlit
cls
call fix_streamlit.bat
goto end

:debug_dashboard
cls
echo Starting Debug Dashboard...
python -m streamlit run debug_dashboard.py
goto end

:end