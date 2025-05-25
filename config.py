"""
WeatherShield AI - Configuration
"""
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # API Keys
    IMGW_API_KEY: str = os.getenv("IMGW_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    
    # Database (optional)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "sqlite:///weathershield.db"  # Use SQLite as fallback
    )
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    
    # Model Parameters
    MODEL_PATH: str = "models/weathershield_v1.pth"
    PREDICTION_HORIZON: int = 3  # hours
    UPDATE_INTERVAL: int = 300  # seconds
    
    # Grid Parameters
    GRID_ZONES: List[str] = field(default_factory=lambda: ["mazowieckie", "malopolskie", "wielkopolskie"])
    CRITICAL_LINES: List[str] = field(default_factory=lambda: ["400kV_North", "400kV_South", "220kV_Central"])
    
    # Alert Thresholds
    WIND_SPEED_WARNING: float = 20.0  # m/s
    WIND_SPEED_CRITICAL: float = 30.0  # m/s
    TEMP_ICE_FORMATION: float = 0.0  # Celsius
    PRECIPITATION_WARNING: float = 50.0  # mm/h
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Kafka Settings (optional)
    KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "")
    KAFKA_TOPIC_WEATHER: str = "weather-data"
    KAFKA_TOPIC_ALERTS: str = "weather-alerts"

config = Config()