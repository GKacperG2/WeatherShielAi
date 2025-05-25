@echo off
echo ========================================
echo   Fixing Streamlit + PyTorch Issues
echo ========================================
echo.

echo Installing compatible Streamlit version...
pip uninstall streamlit -y
pip install streamlit==1.28.1

echo Installing compatible folium versions...
pip uninstall folium streamlit-folium -y
pip install folium==0.14.0 streamlit-folium==0.13.0

echo Installing compatible torch versions...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

echo Clearing Streamlit cache...
rmdir /s /q %USERPROFILE%\.streamlit 2>nul
mkdir %USERPROFILE%\.streamlit 2>nul

echo Creating Streamlit config...
echo [global] > %USERPROFILE%\.streamlit\config.toml
echo developmentMode = false >> %USERPROFILE%\.streamlit\config.toml
echo [server] >> %USERPROFILE%\.streamlit\config.toml
echo enableCORS = false >> %USERPROFILE%\.streamlit\config.toml
echo enableXsrfProtection = false >> %USERPROFILE%\.streamlit\config.toml

echo.
echo ✅ Fixed! Now run: streamlit run dashboard.py
pause
