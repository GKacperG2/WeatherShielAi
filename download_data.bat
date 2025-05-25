@echo off
echo ========================================
echo WeatherShield AI - Data Download
echo ========================================
echo.

REM Check if kaggle is installed
python -m pip show kaggle >nul 2>&1
if errorlevel 1 (
    echo Installing Kaggle API...
    python -m pip install kaggle
)

echo.
echo IMPORTANT: You need Kaggle API credentials!
echo 1. Go to https://www.kaggle.com/account
echo 2. Create New API Token
echo 3. Save kaggle.json to %USERPROFILE%\.kaggle\
echo.
pause

REM Create data directories
mkdir data\weather 2>nul
mkdir data\outages 2>nul
mkdir data\powerplants 2>nul

REM Download datasets
echo.
echo Downloading weather data...
kaggle datasets download -d selfishgene/historical-hourly-weather-data -p data\weather --unzip

echo Downloading power outage data...
kaggle datasets download -d umerhaddii/us-power-outage-data -p data\outages --unzip

echo Downloading power plant data...
kaggle datasets download -d ramjasmaurya/global-powerplants -p data\powerplants --unzip

echo.
echo Download complete!
pause