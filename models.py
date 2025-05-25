"""
WeatherShield AI - Machine Learning Models (Simplified)
"""
import joblib
import pickle
from pathlib import Path
from typing import Dict, Optional
import logging
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class WeatherNowcastLSTM(nn.Module):
    """Simplified LSTM without PyTorch Lightning"""
    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 256,
        output_dim: int = 10,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

class PowerGridGNN(nn.Module):
    def __init__(self, node_features: int = 1, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Simple feedforward layers that work with or without torch-geometric
        self.fc1 = nn.Linear(node_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
       
    def forward(self, x):
        # Handle both batched and single inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class WeatherImpactFusion(nn.Module):
   def __init__(self, weather_dim: int = 10, grid_dim: int = 5, hidden_dim: int = 64):
       super().__init__()
       self.weather_fc = nn.Linear(weather_dim, hidden_dim)
       self.grid_fc = nn.Linear(grid_dim, hidden_dim)
       self.fusion_fc = nn.Linear(hidden_dim * 2, 1)
       
   def forward(self, weather_feat, grid_feat):
       w = F.relu(self.weather_fc(weather_feat))
       g = F.relu(self.grid_fc(grid_feat))
       combined = torch.cat([w, g], dim=1)
       # Remove sigmoid and multiplication by 100 here
       # The sigmoid will be applied in the training function
       return self.fusion_fc(combined)

class WeatherShieldModel:
   """Main model wrapper with proper initialization"""
   
   def __init__(self, model_dir: str = "models"):
       self.model_dir = Path(model_dir)
       self.model_dir.mkdir(exist_ok=True)
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
       # Initialize models
       self.weather_model = None
       self.grid_model = None
       self.fusion_model = None
       self.scaler = StandardScaler()
       self.models_loaded = False
       
       # Try to load existing models with error handling
       try:
           self.load_models()
       except Exception as e:
           logger.error(f"Error during model initialization: {e}")
           # Initialize with dummy models if loading fails
           self._initialize_models()
       
   def load_models(self):
       """Load models with simplified error handling"""
       try:
           # Initialize dummy models first
           self.weather_model = WeatherNowcastLSTM()
           self.grid_model = PowerGridGNN(node_features=1)
           self.fusion_model = WeatherImpactFusion()
           
           # Check if models directory exists
           if not self.model_dir.exists():
               logger.warning(f"Models directory {self.model_dir} does not exist. Using default models.")
               self._finalize_model_setup()
               return
           
           # Try loading weather model - simplified loading
           weather_path = self.model_dir / "weather_nowcast.ckpt"
           if weather_path.exists() and weather_path.stat().st_size > 0:
               try:
                   state_dict = torch.load(weather_path, map_location=self.device, weights_only=True)
                   
                   # Handle both Lightning and regular checkpoints
                   if 'state_dict' in state_dict:
                       model_state = state_dict['state_dict']
                   else:
                       model_state = state_dict
                   
                   # Try to load compatible parameters
                   try:
                       self.weather_model.load_state_dict(model_state, strict=False)
                       logger.info("Weather model loaded successfully")
                   except Exception as e:
                       logger.warning(f"Could not load weather model weights: {e}")
                       
               except Exception as e:
                   logger.warning(f"Failed to load weather model: {e}")
           
           # Try loading grid model
           grid_path = self.model_dir / "grid_gnn.pth"
           if grid_path.exists() and grid_path.stat().st_size > 0:
               try:
                   state_dict = torch.load(grid_path, map_location=self.device, weights_only=True)
                   self.grid_model.load_state_dict(state_dict, strict=False)
                   logger.info("Grid model loaded successfully")
               except Exception as e:
                   logger.warning(f"Failed to load grid model: {e}")
           
           # Try loading fusion model
           fusion_path = self.model_dir / "fusion.pth"
           if fusion_path.exists() and fusion_path.stat().st_size > 0:
               try:
                   state_dict = torch.load(fusion_path, map_location=self.device, weights_only=True)
                   self.fusion_model.load_state_dict(state_dict, strict=False)
                   logger.info("Fusion model loaded successfully")
               except Exception as e:
                   logger.warning(f"Failed to load fusion model: {e}")
           
           # Try loading scaler
           scaler_path = self.model_dir / "scaler.pkl"
           if scaler_path.exists() and scaler_path.stat().st_size > 0:
               try:
                   with open(scaler_path, 'rb') as f:
                       self.scaler = joblib.load(f)
                   logger.info("Scaler loaded successfully")
               except Exception as e:
                   logger.warning(f"Failed to load scaler: {e}")
                   self.scaler = StandardScaler()
           
           self._finalize_model_setup()
           
       except Exception as e:
           logger.error(f"Critical error loading models: {e}")
           self._initialize_models()
   
   def _finalize_model_setup(self):
       """Finalize model setup"""
       # Move models to device
       self.weather_model.to(self.device)
       self.grid_model.to(self.device)
       self.fusion_model.to(self.device)
       
       # Set to eval mode
       self.weather_model.eval()
       self.grid_model.eval()
       self.fusion_model.eval()
       
       # Mark as loaded
       self.models_loaded = True
       logger.info("All models finalized successfully")
   
   def _initialize_models(self):
       """Initialize new models from scratch"""
       self.weather_model = WeatherNowcastLSTM()
       self.grid_model = PowerGridGNN(node_features=1)
       self.fusion_model = WeatherImpactFusion()
       
       # Move to device
       self.weather_model.to(self.device)
       self.grid_model.to(self.device)
       self.fusion_model.to(self.device)
       
       # Set to eval mode
       self.weather_model.eval()
       self.grid_model.eval()
       self.fusion_model.eval()
       
       # Mark as loaded even if not fully trained
       self.models_loaded = True
       
       # Save initial models
       self.save_models()
   
   def save_models(self):
       """Save current models to disk - simplified without Lightning"""
       try:
           # Save weather model as regular PyTorch checkpoint
           torch.save(
               self.weather_model.state_dict(),
               self.model_dir / "weather_nowcast.ckpt"
           )
           
           # Save grid model
           torch.save(
               self.grid_model.state_dict(),
               self.model_dir / "grid_gnn.pth"
           )
           
           # Save fusion model
           torch.save(
               self.fusion_model.state_dict(),
               self.model_dir / "fusion.pth"
           )
           
           # Save scaler
           with open(self.model_dir / "scaler.pkl", 'wb') as f:
               pickle.dump(self.scaler, f)
           
           logger.info("Models saved successfully")
           
       except Exception as e:
           logger.error(f"Error saving models: {e}")
   
   def predict(self, weather_data: pd.DataFrame, grid_state: Optional[Dict] = None) -> Dict:
       """Make predictions with proper error handling"""
       try:
           # Check if models are properly loaded
           if not self.models_loaded:
               logger.warning("Models not properly loaded, using fallback predictions")
               return {
                   'risk_score': 50.0,
                   'weather_forecast': [[0.0] * 10],
                   'confidence': 0.5
               }
           
           with torch.no_grad():
               # Prepare weather data
               weather_tensor = self._prepare_weather_data(weather_data)
               
               # Weather prediction
               weather_pred = self.weather_model(weather_tensor)
               
               # Grid prediction (simplified for now)
               if grid_state:
                   grid_tensor = torch.randn(1, 20).to(self.device)
                   grid_pred = self.grid_model(grid_tensor)
                   
                   # Fusion
                   risk_score = self.fusion_model(weather_pred, grid_pred)
               else:
                   # Simple risk based on weather only
                   risk_score = self._weather_only_risk(weather_pred)
               
               return {
                   'risk_score': float(risk_score.item()),
                   'weather_forecast': weather_pred.cpu().numpy().tolist(),
                   'confidence': self._calculate_confidence(weather_pred)
               }
               
       except Exception as e:
           logger.error(f"Prediction error: {e}")
           # Return default values
           return {
               'risk_score': 50.0,
               'weather_forecast': [[0.0] * 10],
               'confidence': 0.5
           }
   
   def _prepare_weather_data(self, df: pd.DataFrame) -> torch.Tensor:
       """Prepare weather data for model input"""
       # Select and normalize features
       features = ['temperature', 'wind_speed', 'pressure', 'humidity', 'precipitation']
       
       # Ensure all features exist
       for feat in features:
           if feat not in df.columns:
               df[feat] = 0.0
       
       # Get data
       data = df[features].values
       
       # Fit scaler if needed
       if not hasattr(self.scaler, 'mean_'):
           self.scaler.fit(data)
       
       # Scale data
       data_scaled = self.scaler.transform(data)
       
       # Create sequence (using last 24 hours or padding)
       seq_len = 24
       if len(data_scaled) < seq_len:
           # Pad with zeros
           padding = np.zeros((seq_len - len(data_scaled), len(features)))
           data_scaled = np.vstack([padding, data_scaled])
       else:
           data_scaled = data_scaled[-seq_len:]
       
       # Add extra features (time encodings)
       hours = np.arange(seq_len)
       hour_sin = np.sin(2 * np.pi * hours / 24).reshape(-1, 1)
       hour_cos = np.cos(2 * np.pi * hours / 24).reshape(-1, 1)
       
       # Combine features
       full_features = np.hstack([data_scaled, hour_sin, hour_cos])
       
       # Pad to expected input size
       if full_features.shape[1] < 15:
           padding = np.zeros((seq_len, 15 - full_features.shape[1]))
           full_features = np.hstack([full_features, padding])
       
       # Convert to tensor
       tensor = torch.FloatTensor(full_features).unsqueeze(0)
       return tensor.to(self.device)
   
   def _weather_only_risk(self, weather_pred: torch.Tensor) -> torch.Tensor:
       """Calculate risk based on weather prediction only"""
       # Use first few outputs as risk indicators
       wind_risk = torch.sigmoid(weather_pred[0, 0]) * 50
       temp_risk = torch.sigmoid(weather_pred[0, 1]) * 30
       pressure_risk = torch.sigmoid(weather_pred[0, 2]) * 20
       
       combined_risk = wind_risk + temp_risk + pressure_risk
       return torch.clamp(combined_risk, 0, 100)
   
   def _calculate_confidence(self, prediction: torch.Tensor) -> float:
       """Calculate model confidence"""
       # Based on prediction variance
       if prediction.numel() > 1:
           std = torch.std(prediction).item()
           confidence = max(0.3, min(0.95, 1.0 - std / 5.0))
       else:
           confidence = 0.7
       return confidence
   
   def reload(self):
       """Reload models from disk"""
       logger.info("Reloading models...")
       self.load_models()

def create_model(model_dir: str = "models") -> WeatherShieldModel:
   """Create and initialize WeatherShield model with error handling"""
   try:
       return WeatherShieldModel(model_dir)
   except Exception as e:
       logger.error(f"Failed to create model: {e}")
       # Return a minimal working model
       model = WeatherShieldModel.__new__(WeatherShieldModel)
       model.model_dir = Path(model_dir)
       model.device = torch.device("cpu")
       model.weather_model = WeatherNowcastLSTM()
       model.grid_model = PowerGridGNN(node_features=1)
       model.fusion_model = WeatherImpactFusion()
       model.scaler = StandardScaler()
       model.models_loaded = False
       return model