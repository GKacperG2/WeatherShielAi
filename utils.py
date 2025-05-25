"""
WeatherShield AI - Utility Functions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import hashlib
import logging
from datetime import datetime
import os, random, numpy as np, torch


def seed_everything(seed: int = 42) -> None:
    """
    Ustawia deterministyczne ziarno dla Pythona, NumPy i PyTorch
    (CPU + CUDA).  Przydatne do powtarzalnych eksperymentów.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Łagodne deterministyczne tryby CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_wind_chill(temp: float, wind_speed: float) -> float:
    """Calculate wind chill temperature"""
    if temp > 10 or wind_speed < 4.8:
        return temp
    
    return 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + \
           0.3965 * temp * (wind_speed ** 0.16)

def calculate_ice_accumulation(temp: float, precip: float, wind: float) -> float:
    """Estimate ice accumulation on power lines"""
    if temp > 0 or precip == 0:
        return 0.0
    
    # Simplified ice accumulation model
    base_accumulation = precip * 0.1  # mm
    wind_factor = 1 + (wind / 20.0)  # Wind increases accumulation
    temp_factor = max(0, (0 - temp) / 10.0)  # Colder = more ice
    
    return base_accumulation * wind_factor * temp_factor

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points on Earth"""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

@lru_cache(maxsize=1000)
def get_grid_cell(lat: float, lon: float, resolution: float = 0.1) -> str:
    """Convert coordinates to grid cell ID"""
    grid_lat = int(lat / resolution) * resolution
    grid_lon = int(lon / resolution) * resolution
    return f"{grid_lat:.1f},{grid_lon:.1f}"

def interpolate_weather_grid(
    points: List[Dict], 
    grid_resolution: float = 0.1
) -> pd.DataFrame:
    """Interpolate weather data to regular grid"""
    from scipy.interpolate import griddata
    
    # Extract coordinates and values
    lats = [p['lat'] for p in points]
    lons = [p['lon'] for p in points]
    temps = [p['temperature'] for p in points]
    
    # Create grid
    lat_range = np.arange(min(lats), max(lats), grid_resolution)
    lon_range = np.arange(min(lons), max(lons), grid_resolution)
    grid_lats, grid_lons = np.meshgrid(lat_range, lon_range)
    
    # Interpolate
    grid_temps = griddata(
        (lats, lons), temps, (grid_lats, grid_lons), 
        method='cubic'
    )
    
    return pd.DataFrame({
        'lat': grid_lats.flatten(),
        'lon': grid_lons.flatten(),
        'temperature': grid_temps.flatten()
    })

def create_geohash(lat: float, lon: float, precision: int = 6) -> str:
    """Create geohash for spatial indexing"""
    # Simplified geohash implementation
    lat_bin = format(int((lat + 90) * 1000), '020b')
    lon_bin = format(int((lon + 180) * 1000), '020b')
    
    combined = ''.join([lon_bin[i] + lat_bin[i] for i in range(20)])
    return hashlib.md5(combined.encode()).hexdigest()[:precision]

def validate_weather_data(data: Dict) -> Tuple[bool, Optional[str]]:
    """Validate weather data for anomalies"""
    # Temperature checks
    if not -50 <= data.get('temperature', 0) <= 50:
        return False, "Temperature out of range"
    
    # Wind speed checks
    if not 0 <= data.get('wind_speed', 0) <= 100:
        return False, "Wind speed out of range"
    
    # Pressure checks
    if not 900 <= data.get('pressure', 1013) <= 1100:
        return False, "Pressure out of range"
    
    return True, None

def calculate_risk_score(
    weather_data: Dict,
    asset_vulnerability: float = 1.0
) -> float:
    """Calculate combined risk score (0-100)"""
    scores = []
    
    # Wind risk
    wind = weather_data.get('wind_speed', 0)
    if wind > 30:
        wind_score = 100
    elif wind > 20:
        wind_score = 50 + (wind - 20) * 5
    else:
        wind_score = wind * 2.5
    scores.append(wind_score)
    
    # Ice risk
    temp = weather_data.get('temperature', 10)
    precip = weather_data.get('precipitation', 0)
    if temp < 0 and precip > 0:
        ice_score = min(100, 50 + precip * 2)
        scores.append(ice_score)
    
    # Lightning risk
    if weather_data.get('lightning_probability', 0) > 0.5:
        scores.append(80)
    
    # Combined score
    if scores:
        base_score = max(scores)
        return min(100, base_score * asset_vulnerability)
    return 0

def format_alert_message(alert: Dict) -> str:
    """Format alert for notification"""
    severity_emoji = {
        'low': '🟡',
        'medium': '🟠', 
        'high': '🔴',
        'critical': '🚨'
    }
    
    emoji = severity_emoji.get(alert['severity'], '⚠️')
    
    return f"""
{emoji} ALERT: {alert['title']}
Location: {alert['location']}
Risk Score: {alert['risk_score']}/100
Time: {alert['timestamp']}
Actions: {', '.join(alert['recommended_actions'])}
    """.strip()

def create_time_series_features(
    df: pd.DataFrame, 
    lookback: int = 24
) -> pd.DataFrame:
    """Create time series features for ML model"""
    # Lag features
    for col in ['temperature', 'wind_speed', 'pressure']:
        if col in df.columns:
            for lag in [1, 3, 6, 12, 24]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Rolling statistics
    for col in ['temperature', 'wind_speed']:
        if col in df.columns:
            df[f'{col}_roll_mean_6h'] = df[col].rolling(6).mean()
            df[f'{col}_roll_std_6h'] = df[col].rolling(6).std()
            df[f'{col}_roll_max_6h'] = df[col].rolling(6).max()
    
    # Time features
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df

def _prepare_weather_data(self, df: pd.DataFrame) -> torch.Tensor:
    # ...existing code...
    
    # Add missing numpy import for this function
    import numpy as np
    
    # Create sequence (using last 24 hours or padding)
    seq_len = 24
    if len(data_scaled) < seq_len:
        # Pad with zeros
        padding = np.zeros((seq_len - len(data_scaled), len(features)))
        data_scaled = np.vstack([padding, data_scaled])
    else:
        data_scaled = data_scaled[-seq_len:]
    
    # ...existing code...