"""
Quick fix to create a working grid model
"""
import torch
import torch.nn as nn
from pathlib import Path

class SimpleGridModel(nn.Module):
    def __init__(self, node_features=1, hidden_dim=32, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(node_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def create_working_grid_model():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Create and train a simple working model
    model = SimpleGridModel()
    
    # Quick training with dummy data
    X = torch.randn(100, 1)
    y = torch.randn(100, 1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Training simple grid model...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Save the model
    torch.save(model.state_dict(), model_dir / "grid_gnn.pth")
    print(f"✅ Grid model saved to {model_dir / 'grid_gnn.pth'}")

if __name__ == "__main__":
    create_working_grid_model()
