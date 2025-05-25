"""
WeatherShield AI - Data Pipeline
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple
from kafka import KafkaProducer, KafkaConsumer
import redis
from sqlalchemy import create_engine
from config import config
import utils

logger = logging.getLogger(__name__)

class WeatherDataCollector:
    """Collect weather data from multiple sources"""
    
    def __init__(self):
        self.session = None
        self.redis_client = redis.Redis.from_url(config.REDIS_URL)
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def fetch_imgw_data(self) -> List[Dict]:
        """Fetch data from IMGW (Polish meteorological institute)"""
        url = "https://danepubliczne.imgw.pl/api/data/synop"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Transform to standard format
                    transformed = []
                    for station in data:
                        transformed.append({
                            'source': 'imgw',
                            'station_id': station.get('id_stacji'),
                            'station_name': station.get('stacja'),
                            'timestamp': datetime.now().isoformat(),
                            'lat': float(station.get('lat', 0)),
                            'lon': float(station.get('lon', 0)),
                            'temperature': float(station.get('temperatura', 0)),
                            'pressure': float(station.get('cisnienie', 1013)),
                            'wind_speed': float(station.get('predkosc_wiatru', 0)),
                            'wind_direction': float(station.get('kierunek_wiatru', 0)),
                            'humidity': float(station.get('wilgotnosc_wzgledna', 50)),
                            'precipitation': float(station.get('suma_opadu', 0))
                        })
                    
                    return transformed
                    
        except Exception as e:
            logger.error(f"Error fetching IMGW data: {e}")
            return []
    
    async def fetch_openweather_data(self, cities: List[str]) -> List[Dict]:
        """Fetch data from OpenWeatherMap API"""
        if not config.OPENWEATHER_API_KEY:
            logger.warning("OpenWeather API key not configured")
            return []
        
        data = []
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        
        for city in cities:
            params = {
                'q': f"{city},PL",
                'appid': config.OPENWEATHER_API_KEY,
                'units': 'metric'
            }
            
            try:
                async with self.session.get(base_url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        data.append({
                            'source': 'openweather',
                            'station_id': result['id'],
                            'station_name': result['name'],
                            'timestamp': datetime.now().isoformat(),
                            'lat': result['coord']['lat'],
                            'lon': result['coord']['lon'],
                            'temperature': result['main']['temp'],
                            'pressure': result['main']['pressure'],
                            'wind_speed': result['wind']['speed'],
                            'wind_direction': result['wind'].get('deg', 0),
                            'humidity': result['main']['humidity'],
                            'precipitation': result.get('rain', {}).get('1h', 0)
                        })
                        
            except Exception as e:
                logger.error(f"Error fetching OpenWeather data for {city}: {e}")
        
        return data
    
    async def fetch_satellite_data(self) -> Dict:
        """Fetch satellite imagery data (simulated)"""
        # In production, this would connect to Copernicus/Sentinel APIs
        # For demo, we'll simulate satellite-based severe weather detection
        
        return {
            'source': 'satellite',
            'timestamp': datetime.now().isoformat(),
            'cloud_cover_map': np.random.rand(100, 100).tolist(),
            'lightning_detected': np.random.random() > 0.8,
            'storm_cells': [
                {
                    'lat': 52.2 + np.random.normal(0, 0.5),
                    'lon': 21.0 + np.random.normal(0, 0.5),
                    'intensity': np.random.randint(1, 5),
                    'direction': np.random.randint(0, 360),
                    'speed': np.random.randint(10, 50)
                }
                for _ in range(np.random.randint(0, 3))
            ]
        }
    
    async def collect_all_data(self) -> Dict:
        """Collect data from all sources"""
        logger.info("Starting data collection cycle")
        
        # Parallel data collection
        tasks = [
            self.fetch_imgw_data(),
            self.fetch_openweather_data(['Warsaw', 'Krakow', 'Gdansk', 'Wroclaw']),
            self.fetch_satellite_data()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        weather_stations = []
        satellite_data = None
        
        for result in results:
            if isinstance(result, list):
                weather_stations.extend(result)
            elif isinstance(result, dict) and result.get('source') == 'satellite':
                satellite_data = result
            elif isinstance(result, Exception):
                logger.error(f"Data collection error: {result}")
        
        return {
            'weather_stations': weather_stations,
            'satellite': satellite_data,
            'collection_time': datetime.now().isoformat()
        }
    
    def publish_to_kafka(self, data: Dict):
        """Publish data to Kafka for stream processing"""
        try:
            self.kafka_producer.send(
                config.KAFKA_TOPIC_WEATHER,
                value=data
            )
            logger.info("Data published to Kafka")
        except Exception as e:
            logger.error(f"Kafka publish error: {e}")
    
    def cache_latest_data(self, data: Dict):
        """Cache latest data in Redis"""
        try:
            # Store with 5-minute expiry
            self.redis_client.setex(
                'latest_weather_data',
                300,
                json.dumps(data)
            )
            
            # Also store individual station data for quick lookup
            for station in data.get('weather_stations', []):
                key = f"station:{station['station_id']}"
                self.redis_client.setex(key, 300, json.dumps(station))
                
        except Exception as e:
            logger.error(f"Redis cache error: {e}")

class DataProcessor:
    """Process and enrich weather data"""
    
    def __init__(self):
        self.engine = create_engine(config.DATABASE_URL)
        
    def process_weather_batch(self, data: Dict) -> pd.DataFrame:
        """Process batch of weather data"""
        df = pd.DataFrame(data['weather_stations'])
        
        if df.empty:
            return df
        
        # Add derived features
        df['wind_chill'] = df.apply(
            lambda row: utils.calculate_wind_chill(
                row['temperature'], row['wind_speed']
            ), axis=1
        )
        
        df['ice_risk'] = df.apply(
            lambda row: utils.calculate_ice_accumulation(
                row['temperature'], row['precipitation'], row['wind_speed']
            ), axis=1
        )
        
        # Add grid cell
        df['grid_cell'] = df.apply(
            lambda row: utils.get_grid_cell(row['lat'], row['lon']),
            axis=1
        )
        
        # Add time features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalous weather patterns"""
        # Z-score based anomaly detection
        numeric_cols = ['temperature', 'wind_speed', 'pressure']
        
        for col in numeric_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df[f'{col}_zscore'] = (df[col] - mean) / (std + 1e-6)
                df[f'{col}_anomaly'] = abs(df[f'{col}_zscore']) > 3
        
        # Combined anomaly score
        anomaly_cols = [col for col in df.columns if col.endswith('_anomaly')]
        df['anomaly_score'] = df[anomaly_cols].sum(axis=1)
        
        return df
    
    def save_to_database(self, df: pd.DataFrame):
        """Save processed data to database"""
        try:
            df.to_sql(
                'weather_observations',
                self.engine,
                if_exists='append',
                index=False
            )
            logger.info(f"Saved {len(df)} records to database")
        except Exception as e:
            logger.error(f"Database save error: {e}")

class StreamProcessor:
    """Real-time stream processing with Kafka"""
    
    def __init__(self):
        self.consumer = KafkaConsumer(
            config.KAFKA_TOPIC_WEATHER,
            bootstrap_servers=config.KAFKA_BROKER,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='weather-processor-group'
        )
        self.producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.processor = DataProcessor()
        
    def process_stream(self):
        """Process incoming weather data stream"""
        logger.info("Starting stream processor")
        
        for message in self.consumer:
            try:
                data = message.value
                
                # Process data
                df = self.processor.process_weather_batch(data)
                df = self.processor.detect_anomalies(df)
                
                # Check for alerts
                alerts = self._check_alert_conditions(df)
                
                # Publish alerts
                for alert in alerts:
                    self.producer.send(
                        config.KAFKA_TOPIC_ALERTS,
                        value=alert
                    )
                
                # Save to database
                if not df.empty:
                    self.processor.save_to_database(df)
                    
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
    
    def _check_alert_conditions(self, df: pd.DataFrame) -> List[Dict]:
        """Check for alert conditions in weather data"""
        alerts = []
        
        # High wind speed alerts
        high_wind = df[df['wind_speed'] > config.WIND_SPEED_WARNING]
        for _, row in high_wind.iterrows():
            severity = 'critical' if row['wind_speed'] > config.WIND_SPEED_CRITICAL else 'warning'
            
            alerts.append({
                'type': 'high_wind',
                'severity': severity,
                'location': row['station_name'],
                'lat': row['lat'],
                'lon': row['lon'],
                'value': row['wind_speed'],
                'threshold': config.WIND_SPEED_WARNING,
                'timestamp': row['timestamp'].isoformat(),
                'message': f"High wind speed detected: {row['wind_speed']:.1f} m/s"
            })
        
        # Ice formation risk
        ice_risk = df[(df['temperature'] < config.TEMP_ICE_FORMATION) & 
                      (df['precipitation'] > 0)]
        for _, row in ice_risk.iterrows():
            alerts.append({
                'type': 'ice_risk',
                'severity': 'warning',
                'location': row['station_name'],
                'lat': row['lat'],
                'lon': row['lon'],
                'value': row['ice_risk'],
                'timestamp': row['timestamp'].isoformat(),
                'message': f"Ice formation risk: {row['ice_risk']:.2f} mm"
            })
        
        # Anomaly alerts
        anomalies = df[df['anomaly_score'] > 2]
        for _, row in anomalies.iterrows():
            alerts.append({
                'type': 'anomaly',
                'severity': 'info',
                'location': row['station_name'],
                'lat': row['lat'],
                'lon': row['lon'],
                'value': row['anomaly_score'],
                'timestamp': row['timestamp'].isoformat(),
                'message': f"Unusual weather pattern detected"
            })
        
        return alerts

# Async data collection runner
async def run_data_collection():
    """Run continuous data collection"""
    async with WeatherDataCollector() as collector:
        while True:
            try:
                # Collect data
                data = await collector.collect_all_data()
                
                # Publish to Kafka
                collector.publish_to_kafka(data)
                
                # Cache in Redis
                collector.cache_latest_data(data)
                
                # Wait for next cycle
                await asyncio.sleep(config.UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Data collection error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute

if __name__ == "__main__":
    # Run data collection
    asyncio.run(run_data_collection())