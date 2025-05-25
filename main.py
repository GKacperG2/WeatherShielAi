"""
WeatherShield AI - Main Application Entry Point (Fixed)
"""
import asyncio
import logging
import signal
import sys
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config
from digital_twin import create_digital_twin
from models import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
model = None
digital_twin = None

# Pydantic models
class WeatherDataRequest(BaseModel):
    lat: float
    lon: float
    radius: Optional[float] = 50.0

class PredictionResponse(BaseModel):
    risk_score: float
    confidence: float
    weather_forecast: Dict
    grid_impact: Optional[Dict]
    recommendations: List[Dict]
    timestamp: str

class AlertResponse(BaseModel):
    alerts: List[Dict]
    total_count: int
    critical_count: int
    timestamp: str

# Lifespan context manager (modern FastAPI approach)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, digital_twin
    
    logger.info("▶ Starting WeatherShield AI services...")
    
    try:
        model = create_model()
        digital_twin = create_digital_twin()
        logger.info("✅ All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down WeatherShield AI")

# Create FastAPI app with lifespan
app = FastAPI(
    title="WeatherShield AI API",
    description="Real-time weather impact analysis for power grids",
    version="1.2.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "WeatherShield AI",
        "status": "operational",
        "version": "1.2.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    status = {
        "status": "healthy",
        "services": {
            "model": model is not None,
            "digital_twin": digital_twin is not None
        },
        "timestamp": datetime.now().isoformat()
    }
    
    if not all(status["services"].values()):
        raise HTTPException(status_code=503, detail="Some services are not available")
    
    return status

@app.post("/predict", response_model=PredictionResponse)
async def predict_impact(request: WeatherDataRequest):
    """Predict weather impact on power grid"""
    try:
        # Generate sample weather data for the location
        weather_data = pd.DataFrame([{
            'lat': request.lat,
            'lon': request.lon,
            'temperature': 10.0,
            'wind_speed': 15.0,
            'pressure': 1013.0,
            'humidity': 70.0,
            'precipitation': 0.0,
            'timestamp': datetime.now()
        }])
        
        # Get model prediction
        prediction = model.predict(weather_data)
        
        # Get grid impact
        impact_scores = digital_twin.calculate_impact_scores(weather_data)
        recommendations = digital_twin.recommend_actions(impact_scores)
        
        return PredictionResponse(
            risk_score=prediction['risk_score'],
            confidence=prediction['confidence'],
            weather_forecast=prediction['weather_forecast'],
            grid_impact=digital_twin.export_state(),
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts", response_model=AlertResponse)
async def get_active_alerts():
    """Get current active alerts"""
    try:
        # Generate sample alerts for demo
        alerts = [
            {
                "id": "alert_001",
                "type": "high_wind",
                "severity": "high",
                "location": "Warsaw Region",
                "lat": 52.2297,
                "lon": 21.0122,
                "risk_score": 75,
                "message": "Wind speed exceeding safety threshold",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        critical_count = sum(1 for a in alerts if a['severity'] == 'critical')
        
        return AlertResponse(
            alerts=alerts,
            total_count=len(alerts),
            critical_count=critical_count,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Alert fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
async def simulate_scenario(
    scenario: str,
    severity: int = 5,
    duration_hours: int = 6
):
    """Simulate weather scenario"""
    scenarios = {
        "ice_storm": {
            "temperature": -5,
            "precipitation": 20,
            "wind_speed": 15
        },
        "high_wind": {
            "temperature": 15,
            "precipitation": 0,
            "wind_speed": 35
        },
        "heat_wave": {
            "temperature": 38,
            "precipitation": 0,
            "wind_speed": 5
        }
    }
    
    if scenario not in scenarios:
        raise HTTPException(status_code=400, detail="Invalid scenario")
    
    return {
        "scenario": scenario,
        "severity": severity,
        "duration_hours": duration_hours,
        "impact_summary": {
            "affected_assets": severity * 3,
            "estimated_downtime": duration_hours * severity / 5,
            "risk_score": min(100, severity * 15)
        },
        "timestamp": datetime.now().isoformat()
    }

def run_api_server():
    """Run the FastAPI server"""
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Shutting down WeatherShield AI...")
    sys.exit(0)

def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting WeatherShield AI System")
    logger.info(f"API will be available at http://{config.API_HOST}:{config.API_PORT}")
    
    # Start API server
    run_api_server()

if __name__ == "__main__":
    main()