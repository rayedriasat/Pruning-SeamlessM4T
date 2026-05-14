import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

class FeatureDataset(Dataset):
    def __init__(self, split_dir):
        self.feature_files = sorted(glob.glob(os.path.join(split_dir, "*_features.pt")))
        
    def __len__(self):
        return len(self.feature_files)
        
    def __getitem__(self, idx):
        feat_path = self.feature_files[idx]
        label_path = feat_path.replace("_features.pt", "_labels.pt")
        
        features = torch.load(feat_path) # [T, 1024]
        labels = torch.load(label_path).unsqueeze(-1) # [T, 1]
        return features, labels

# --- THE FIX: Custom collation to pad variable-length sequences ---
def pad_collate_fn(batch):
    # Separate features and labels
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad sequences to the max length in this batch. 
    # padding_value=0.0 represents silence, which aligns perfectly with our labels!
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0.0)
    
    return padded_features, padded_labels

class CIFBoundaryDetector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Using your updated paths
    train_dir = "/home/nihal/CSE465/Duplex_making/working_dataset/train"
    val_dir = "/home/nihal/CSE465/Duplex_making/working_dataset/val"
    
    # Pass the collate_fn into the DataLoaders
    train_loader = DataLoader(FeatureDataset(train_dir), batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(FeatureDataset(val_dir), batch_size=16, shuffle=False, collate_fn=pad_collate_fn)
    
    model = CIFBoundaryDetector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features.float())
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
    torch.save(model.state_dict(), "boundary_adapter.pt")
    print("Model saved to boundary_adapter.pt")

if __name__ == "__main__":
    train()