"""
WeatherShield AI - Digital Twin of Power Grid (Fixed for PyPSA 0.26+)
"""
import pypsa
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from dataclasses import dataclass

from config import config
import utils

logger = logging.getLogger(__name__)

@dataclass
class GridAsset:
    """Representation of a grid asset"""
    asset_id: str
    asset_type: str
    lat: float
    lon: float
    voltage_level: int
    capacity: float
    weather_vulnerability: float

class DigitalTwin:
    """Digital twin of the power grid"""
    
    def __init__(self):
        self.network = pypsa.Network()
        self.assets = {}
        self.vulnerability_map = {}
        self._initialize_grid()
        
    def _initialize_grid(self):
        """Initialize power grid model"""
        # Set up time series with lowercase 'h'
        self.network.set_snapshots(pd.date_range('2024-01-01', periods=24, freq='h'))
        
        # Add buses using the new API
        buses_data = {
            'x': [21.0, 19.5, 20.0, 22.0, 18.5],
            'y': [52.2, 50.0, 51.5, 53.0, 51.0],
            'v_nom': [400, 400, 220, 220, 110],
            'carrier': ['AC'] * 5
        }
        self.network.add(
            "Bus",
            ['Warsaw', 'Krakow', 'Lodz', 'Gdansk', 'Wroclaw'],
            **buses_data
        )
        
        # Add generators
        generators_data = {
            'bus': ['Warsaw', 'Krakow', 'Gdansk'],
            'p_nom': [1000, 800, 600],
            'marginal_cost': [50, 45, 55],
            'carrier': ['coal', 'gas', 'coal']
        }
        self.network.add(
            "Generator",
            ['Gen_Warsaw', 'Gen_Krakow', 'Gen_Gdansk'],
            **generators_data
        )
        
        # Add wind farms
        wind_data = {
            'bus': ['Gdansk', 'Wroclaw'],
            'p_nom': [200, 150],
            'carrier': ['wind'] * 2,
            'marginal_cost': [0, 0]
        }
        self.network.add(
            "Generator",
            ['Wind_North', 'Wind_West'],
            **wind_data
        )
        
        # Add transmission lines
        lines_data = {
            'bus0': ['Warsaw', 'Warsaw', 'Krakow', 'Lodz', 'Gdansk'],
            'bus1': ['Krakow', 'Lodz', 'Wroclaw', 'Gdansk', 'Wroclaw'],
            'x': [0.1, 0.08, 0.12, 0.09, 0.15],
            'r': [0.01, 0.008, 0.012, 0.009, 0.015],
            's_nom': [1000, 1200, 800, 900, 600],
            'length': [300, 150, 250, 200, 400]
        }
        self.network.add(
            "Line",
            ['L1', 'L2', 'L3', 'L4', 'L5'],
            **lines_data
        )
        
        # Add loads
        loads_data = {
            'bus': ['Warsaw', 'Krakow', 'Lodz', 'Gdansk', 'Wroclaw'],
            'p_set': [800, 600, 400, 500, 450]
        }
        self.network.add(
            "Load",
            ['Load_Warsaw', 'Load_Krakow', 'Load_Lodz', 'Load_Gdansk', 'Load_Wroclaw'],
            **loads_data
        )
        
        # Initialize assets
        self._initialize_assets()
        
    def _initialize_assets(self):
        """Initialize asset registry with vulnerability scores"""
        # Lines
        for idx, line in self.network.lines.iterrows():
            bus0 = self.network.buses.loc[line.bus0]
            bus1 = self.network.buses.loc[line.bus1]
            
            self.assets[idx] = GridAsset(
                asset_id=idx,
                asset_type='line',
                lat=(bus0.y + bus1.y) / 2,
                lon=(bus0.x + bus1.x) / 2,
                voltage_level=int(max(bus0.v_nom, bus1.v_nom)),
                capacity=line.s_nom,
                weather_vulnerability=self._calculate_line_vulnerability(line)
            )
        
        # Generators
        for idx, gen in self.network.generators.iterrows():
            bus = self.network.buses.loc[gen.bus]
            
            self.assets[idx] = GridAsset(
                asset_id=idx,
                asset_type='generator',
                lat=bus.y,
                lon=bus.x,
                voltage_level=int(bus.v_nom),
                capacity=gen.p_nom,
                weather_vulnerability=0.3 if gen.carrier == 'wind' else 0.1
            )
    
    def _calculate_line_vulnerability(self, line) -> float:
        """Calculate weather vulnerability for transmission line"""
        length_factor = min(1.0, line.length / 500)
        voltage_factor = 0.5 if line.bus0 in ['Warsaw', 'Krakow'] else 0.3
        return length_factor * 0.7 + voltage_factor * 0.3
    
    def calculate_impact_scores(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate impact scores for all assets"""
        impact_scores = []
        
        for asset_id, asset in self.assets.items():
            # Simple risk calculation
            base_risk = 30.0
            
            # Add weather-based risk
            if not weather_data.empty:
                avg_wind = weather_data['wind_speed'].mean()
                wind_risk = min(50, avg_wind * 2)
                base_risk += wind_risk
            
            impact_scores.append({
                'asset_id': asset_id,
                'asset_type': asset.asset_type,
                'lat': asset.lat,
                'lon': asset.lon,
                'weather_risk': base_risk,
                'grid_factor': 1.0,
                'impact_score': min(100, base_risk * asset.weather_vulnerability),
                'timestamp': datetime.now()
            })
        
        return pd.DataFrame(impact_scores)
    
    def recommend_actions(self, impact_scores: pd.DataFrame) -> List[Dict]:
        """Recommend operational actions based on impact scores"""
        recommendations = []
        
        # High risk assets
        high_risk = impact_scores[impact_scores['impact_score'] > 70]
        
        for _, asset in high_risk.iterrows():
            if asset['asset_type'] == 'line':
                recommendations.append({
                    'asset_id': asset['asset_id'],
                    'action': 'reduce_loading',
                    'priority': 'high',
                    'description': f"Reduce loading on line {asset['asset_id']} to 70% capacity",
                    'reason': f"High weather risk (score: {asset['impact_score']:.1f})"
                })
        
        return recommendations
    
    def export_state(self) -> Dict:
        """Export current grid state for visualization"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'buses': [],
            'lines': [],
            'generators': [],
            'loads': []
        }
        
        # Export buses
        for idx, bus in self.network.buses.iterrows():
            state['buses'].append({
                'id': idx,
                'lat': bus.y,
                'lon': bus.x,
                'voltage': bus.v_nom
            })
        
        # Export lines
        for idx, line in self.network.lines.iterrows():
            state['lines'].append({
                'id': idx,
                'from': line.bus0,
                'to': line.bus1,
                'loading': 0.5 + np.random.random() * 0.4,  # Simulated loading
                'capacity': line.s_nom,
                'status': 'normal'
            })
        
        # Export generators
        for idx, gen in self.network.generators.iterrows():
            state['generators'].append({
                'id': idx,
                'bus': gen.bus,
                'type': gen.carrier,
                'capacity': gen.p_nom,
                'output': gen.p_nom * 0.7
            })
        
        return state

def create_digital_twin() -> DigitalTwin:
    """Factory function to create digital twin instance"""
    return DigitalTwin()