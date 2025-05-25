"""
Debug Helper do znajdowania i naprawiania błędów w dashboard.py
"""
import os
import re

file_path = "dashboard.py"

def analyze_python_file(file_path):
    """Analizuje plik Python pod kątem potencjalnych błędów składni i logiki."""
    print(f"\n===== ANALIZA PLIKU {file_path} =====")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Sprawdź podstawową składnię
        try:
            compile(content, file_path, 'exec')
            print("✅ Podstawowa składnia Pythona: OK")
        except SyntaxError as e:
            line_no = e.lineno
            print(f"❌ Błąd składni w linii {line_no}: {e}")
            
            # Pokaż linię z błędem i kontekst
            lines = content.split('\n')
            start = max(0, line_no - 3)
            end = min(len(lines), line_no + 2)
            
            print("\nKontekst błędu:")
            for i in range(start, end):
                prefix = "→ " if i + 1 == line_no else "  "
                print(f"{prefix}{i + 1}: {lines[i]}")
        
        # Sprawdź inne typowe problemy
        check_for_problems(content)
        
    except Exception as e:
        print(f"❌ Błąd podczas analizy pliku: {str(e)}")

def check_for_problems(content):
    """Sprawdza typowe problemy w kodzie."""
    issues = []
    
    # Niezamknięte cudzysłowy
    quotes = ["'", '"', '"""', "'''"]
    for q in quotes:
        if content.count(q) % 2 != 0:
            issues.append(f"⚠️ Możliwy problem: Niezamknięty cudzysłów {q}")
    
    # Brakujące nawiasy
    brackets = ['()', '[]', '{}']
    for b in brackets:
        if content.count(b[0]) != content.count(b[1]):
            issues.append(f"⚠️ Możliwy problem: Niezbalansowane nawiasy {b}")
    
    # Mieszanie spacji i tabulatorów
    if '\t' in content and '    ' in content:
        issues.append("⚠️ Możliwy problem: Mieszanie tabulatorów i spacji do wcięć")
    
    # Sprawdź f-stringi bez cudzysłowu zamykającego
    fstrings = re.findall(r'f["\'].*$', content, re.MULTILINE)
    if fstrings:
        issues.append(f"⚠️ Możliwy problem: Niezamknięte f-stringi: {len(fstrings)} wystąpień")
    
    # Sprawdź zagubione fragmenty kodu (linie, które wyglądają jak urwane)
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith('#') and not (line.endswith(':') or line.endswith(',') or line.endswith(')') or line.endswith('}') or line.endswith(']') or line.endswith('"') or line.endswith("'") or line.endswith('\\') or line.endswith(';')):
            if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith('#') and not lines[i + 1].strip().startswith('.') and not lines[i + 1].strip().startswith('else') and not lines[i + 1].strip().startswith('elif'):
                issues.append(f"⚠️ Możliwy problem: Urwany fragment kodu w linii {i + 1}: '{line}'")
    
    if issues:
        print("\nZnaleziono potencjalne problemy:")
        for issue in issues:
            print(issue)
    else:
        print("✅ Nie znaleziono innych potencjalnych problemów")

def create_fixed_file(file_path):
    """Tworzy naprawioną wersję pliku."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Spróbuj naprawić najpopularniejsze problemy
        fixed_content = content
        
        # Napraw linie plotly które zostały pomieszane
        problematic_pattern = r'fig\.update_layout\(\s*tyle="open-street-map",'
        if re.search(problematic_pattern, fixed_content):
            fixed_content = re.sub(
                problematic_pattern, 
                'fig.update_layout(\n        mapbox_style="carto-darkmatter",', 
                fixed_content
            )
        
        # Napraw importy plotly jeśli są pomieszane
        if "import plotly.express as px" in fixed_content and "fig = px.scatter_mapbox(" in fixed_content:
            if "import plotly.express as px" not in fixed_content.split("fig = px.scatter_mapbox(")[0]:
                fixed_content = fixed_content.replace(
                    "# Użyj alternatywnej mapki Plotly\n                st.markdown(\"### 🗺️ Alternatywna Mapa Pogodowa\")",
                    "# Użyj alternatywnej mapki Plotly\n                st.markdown(\"### 🗺️ Alternatywna Mapa Pogodowa\")\n                import plotly.express as px"
                )
        
        # Napraw mieszane linie kodu
        fixed_content = re.sub(
            r'lat="lat",emu"\)',
            'lat="lat",',
            fixed_content
        )
        
        fixed_content = re.sub(
            r'lon="lon",.*hover_name="station_name",.*hover_data=',
            'lon="lon",\n                        hover_name="station_name",\n                        hover_data=',
            fixed_content
        )
        
        # Zapisz naprawiony plik
        backup_path = file_path + ".backup"
        fixed_path = file_path.replace(".py", "_fixed.py")
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        with open(fixed_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"\n✅ Utworzono kopię zapasową: {backup_path}")
        print(f"✅ Utworzono naprawioną wersję: {fixed_path}")
        print(f"Uruchom z naprawioną wersją: streamlit run {fixed_path}")
        
    except Exception as e:
        print(f"❌ Błąd podczas naprawiania pliku: {str(e)}")

if __name__ == "__main__":
    if not os.path.exists(file_path):
        print(f"Plik {file_path} nie istnieje!")
    else:
        analyze_python_file(file_path)
        create_fixed_file(file_path)
