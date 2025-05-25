"""Quick fix to ensure all models exist"""
import torch
from pathlib import Path
from models import WeatherNowcastLSTM, PowerGridGNN, WeatherImpactFusion
from sklearn.preprocessing import StandardScaler
import pickle

def create_dummy_models():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Create weather model
    if not (model_dir / "weather_nowcast.ckpt").exists():
        model = WeatherNowcastLSTM()
        torch.save(model.state_dict(), model_dir / "weather_nowcast.ckpt")
        print("✅ Created weather model")
    
    # Create grid model  
    if not (model_dir / "grid_gnn.pth").exists():
        model = PowerGridGNN(node_features=1)
        torch.save(model.state_dict(), model_dir / "grid_gnn.pth")
        print("✅ Created grid model")
    
    # Create fusion model
    if not (model_dir / "fusion.pth").exists():
        model = WeatherImpactFusion()
        torch.save(model.state_dict(), model_dir / "fusion.pth")
        print("✅ Created fusion model")
    
    # Create scaler
    if not (model_dir / "scaler.pkl").exists():
        scaler = StandardScaler()
        with open(model_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ Created scaler")
    
    print("🎉 All models are now available!")

if __name__ == "__main__":
    create_dummy_models()
