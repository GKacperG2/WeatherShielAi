@echo off
echo ========================================
echo WeatherShield AI - Demo Mode
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH!
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing required packages...
    python -m pip install streamlit streamlit-folium pandas numpy plotly folium torch scikit-learn pypsa
)

REM Start dashboard using python -m
echo.
echo Starting WeatherShield AI Dashboard...
echo Dashboard will open at: http://localhost:8501
echo.
python -m streamlit run dashboard.py

pause