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

def pad_collate_fn(batch):
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
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

def calculate_accuracy(outputs, labels):
    # Convert probabilities to binary predictions (0 or 1)
    predictions = (outputs >= 0.5).float()
    # Count how many match the true labels
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    return correct, total

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define all three paths
    train_dir = "/home/nihal/CSE465/Duplex_making/working_dataset/train"
    val_dir = "/home/nihal/CSE465/Duplex_making/working_dataset/val"
    test_dir = "/home/nihal/CSE465/Duplex_making/working_dataset/test"
    
    # DataLoaders for all three splits
    train_loader = DataLoader(FeatureDataset(train_dir), batch_size=16, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(FeatureDataset(val_dir), batch_size=16, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(FeatureDataset(test_dir), batch_size=16, shuffle=False, collate_fn=pad_collate_fn)
    
    model = CIFBoundaryDetector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    epochs = 10
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate Training Accuracy
            correct, total = calculate_accuracy(outputs, labels)
            train_correct += correct
            train_total += total
            
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features.float())
                loss = criterion(outputs, labels.float())
                
                val_loss += loss.item()
                
                # Calculate Validation Accuracy
                correct, total = calculate_accuracy(outputs, labels)
                val_correct += correct
                val_total += total
                
        # Calculate Epoch Averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = (train_correct / train_total) * 100
        val_acc = (val_correct / val_total) * 100
                
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} - Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} - Acc: {val_acc:.2f}%")
        
    torch.save(model.state_dict(), "boundary_adapter.pt")
    print("\nModel saved to boundary_adapter.pt")
    
    # --- TEST EVALUATION ---
    print("\nRunning final evaluation on Test Set...")
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features.float())
            loss = criterion(outputs, labels.float())
            
            test_loss += loss.item()
            
            correct, total = calculate_accuracy(outputs, labels)
            test_correct += correct
            test_total += total
            
    avg_test_loss = test_loss / len(test_loader)
    test_acc = (test_correct / test_total) * 100
    
    print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%\n")

if __name__ == "__main__":
    train()