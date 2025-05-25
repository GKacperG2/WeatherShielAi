"""
Minimalna wersja dashboard do debugowania
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="WeatherShield AI - Debug",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ WeatherShield AI - Test Dashboard")

# Test podstawowych funkcji
st.header("🔧 Test Podstawowych Funkcji")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Status", "✅ OK")

with col2:
    st.metric("Test Pandas", "✅ OK" if pd.__version__ else "❌ Error")

with col3:
    st.metric("Test NumPy", "✅ OK" if np.__version__ else "❌ Error")

# Test importów
st.header("📦 Test Importów")

import_status = {}

try:
    from config import config
    import_status["config"] = "✅ OK"
except Exception as e:
    import_status["config"] = f"❌ {str(e)}"

try:
    from models import create_model
    import_status["models"] = "✅ OK"
except Exception as e:
    import_status["models"] = f"❌ {str(e)}"

try:
    from digital_twin import create_digital_twin
    import_status["digital_twin"] = "✅ OK"
except Exception as e:
    import_status["digital_twin"] = f"❌ {str(e)}"

try:
    import utils
    import_status["utils"] = "✅ OK"
except Exception as e:
    import_status["utils"] = f"❌ {str(e)}"

for module, status in import_status.items():
    st.write(f"**{module}**: {status}")

# Test danych
st.header("📊 Test Generowania Danych")

try:
    data = pd.DataFrame({
        'x': np.arange(10),
        'y': np.random.randn(10)
    })
    st.line_chart(data.set_index('x'))
    st.success("✅ Generowanie danych działa")
except Exception as e:
    st.error(f"❌ Błąd generowania danych: {str(e)}")

# Test mapy
st.header("🗺️ Test Mapy")

try:
    import folium
    st.success("✅ Folium załadowany")
    
    # Test prostej mapy
    m = folium.Map(location=[52.0, 19.0], zoom_start=6)
    folium.Marker([52.2297, 21.0122], popup="Warszawa").add_to(m)
    
    from streamlit_folium import st_folium
    st_folium(m, width=700, height=300, key="test_map")
    st.success("✅ Prosta mapa działa")
    
except Exception as e:
    st.error(f"❌ Błąd mapy: {str(e)}")

# Test cache
st.header("🔄 Test Cache")

try:
    @st.cache_data
    def test_cache_function():
        return {"test": "data", "number": 123}
    
    result = test_cache_function()
    st.write(f"Cache test: {result}")
    st.success("✅ Cache działa")
    
except Exception as e:
    st.error(f"❌ Błąd cache: {str(e)}")

# Test syntaxy głównego dashboard
st.header("🔍 Test Składni Dashboard")

try:
    # Próba importu głównego dashboard
    import importlib.util
    spec = importlib.util.spec_from_file_location("dashboard", "dashboard.py")
    if spec and spec.loader:
        dashboard_module = importlib.util.module_from_spec(spec)
        # Nie ładujemy modułu, tylko sprawdzamy składnię
        st.success("✅ Składnia dashboard.py jest poprawna")
    else:
        st.error("❌ Nie można załadować dashboard.py")
        
except SyntaxError as e:
    st.error(f"❌ Błąd składni w dashboard.py: {str(e)}")
    st.code(f"Linia {e.lineno}: {e.text if e.text else 'Brak tekstu'}")
    
except Exception as e:
    st.error(f"❌ Inny błąd dashboard.py: {str(e)}")

# Test funkcji symulacji
st.header("🎮 Test Centrum Symulacji")

if st.button("🚀 Test Mini Symulacji"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(5):
        progress_bar.progress((i + 1) / 5)
        status_text.text(f"Krok {i + 1}/5 - Test animacji")
        
        # Symuluj dane pogodowe
        test_data = pd.DataFrame({
            'station_name': ['Test_Station'],
            'lat': [52.0],
            'lon': [19.0],
            'wind_speed': [15 + i * 5],
            'temperature': [20 - i],
            'precipitation': [i * 2]
        })
        
        st.dataframe(test_data, use_container_width=True)
        
        import time
        time.sleep(1)
    
    st.success("✅ Test symulacji zakończony pomyślnie!")

st.info("🔄 Jeśli wszystkie testy przeszły pomyślnie, główny dashboard powinien działać")

# Diagnostyka błędów
st.header("🩺 Diagnostyka Błędów")

st.markdown("""
**Najczęstsze problemy:**

1. **SyntaxError**: Sprawdź czy wszystkie cudzysłowy są zamknięte
2. **ImportError**: Sprawdź czy wszystkie pakiety są zainstalowane
3. **Folium Error**: Uruchom `pip install folium==0.14.0 streamlit-folium==0.13.0`
4. **Model Error**: Uruchom `python training_pipeline.py` aby utworzyć modele

**Jeśli nadal są problemy:**
- Sprawdź logs w cmd/terminal
- Uruchom `weathershield.bat` opcja 6 (Fix Streamlit)
- Sprawdź czy Python wersja >= 3.8
""")
