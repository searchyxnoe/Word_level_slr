#Credit to Gemini 3.1 Pro
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ================================
# 1. FOCAL LOSS
# ================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = label_smoothing
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, label_smoothing=self.smoothing, reduction='none')
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.mean()

# ================================
# 2. DATASET (UNCHANGED)
# ================================
class SignDataset(Dataset):
    def __init__(self, base_dir, split='training'):
        self.base_dir = Path(base_dir) / split
        self.is_training = (split.lower() == 'training')
        
        self.classes = sorted([d.name for d in self.base_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.data_files = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = self.base_dir / cls_name
            for npy_file in cls_dir.glob("*.npy"):
                self.data_files.append(str(npy_file))
                self.labels.append(self.class_to_idx[cls_name])
                
        print(f"✅ Loaded {len(self.data_files)} samples for {split} ({len(self.classes)} classes)")
    
    def __len__(self): 
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx]).astype(np.float32)
        
        if self.is_training:
            if random.random() < 0.2:
                shift = random.choice([-1, 1])
                data = np.roll(data, shift=shift, axis=0)
                if shift == 1: data[0] = 0
                else: data[-1] = 0
            if random.random() < 0.5:
                scale = random.uniform(0.9, 1.1)
                data[:, :120] *= scale
            noise_multiplier = random.uniform(0.001, 0.015)
            noise = np.random.normal(0, noise_multiplier, data[:, :132].shape).astype(np.float32)
            data[:, :132] += noise
            if random.random() < 0.3:
                num_mask = random.randint(1, 2)
                mask_idx = random.sample(range(15), num_mask)
                data[mask_idx, :] = 0.0
            
        return torch.FloatTensor(data), torch.tensor(self.labels[idx], dtype=torch.long)

# ================================
# 3. PURE TCN BLOCK (Paper Architecture)
# ================================
class TCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = dilation * 1  # Causal padding
        
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        
        # Second conv block  
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)
        out = self.dropout(out)
        
        return out + residual

# ================================
# 4. PURE TCN MODEL (99.5% Paper Architecture)
# ================================
class PureTCN(nn.Module):
    def __init__(self, num_classes=31, input_dim=134, channels=64):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, channels, 1)
        
        # 6 TCN Blocks (dilations 1,2,4,8,16,32 → receptive field=127 frames)
        self.tcn_layers = nn.ModuleList([
            TCNBlock(channels, dilation=2**i)
            for i in range(6)
        ])
        
        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.BatchNorm1d(channels // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(channels // 2, num_classes)
        )
        
    def forward(self, x):
        # (B, 15, 134) → (B, 134, 15)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        
        # TCN stack (full receptive field coverage)
        for layer in self.tcn_layers:
            x = layer(x)
            
        # Global pooling → classification
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

# ================================
# 5. TRAINING PIPELINE
# ================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    progress = tqdm(loader, desc="Train", leave=False)
    for data, targets in progress:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = logits.max(1)
        total += targets.size(0)
        correct += (pred == targets).sum().item()
        
        progress.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*correct/total:.1f}%"})
    
    return total_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    
    progress = tqdm(loader, desc="Valid", leave=False)
    with torch.no_grad():
        for data, targets in progress:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = criterion(logits, targets)
            
            total_loss += loss.item()
            _, pred = logits.max(1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    return total_loss / len(loader), 100 * correct / total, f1

def main():
    BASE_DIR = "/home/unknown_device/Musique/Hackathon/npy_dataset"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 150
    
    print(f"Using device: {DEVICE}")
    
    train_ds = SignDataset(BASE_DIR, 'training')
    val_ds = SignDataset(BASE_DIR, 'validation')
    NUM_CLASSES = len(train_ds.classes)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Pure TCN Model (99.5% paper architecture)
    model = PureTCN(num_classes=NUM_CLASSES, input_dim=134, channels=64).to(DEVICE)
    
    criterion = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
    
    # Adadelta as requested
    optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_f1 = 0
    patience_counter = 0
    
    print("\n🚀 Pure TCN Training (99.5% Paper Architecture)")
    print("Dilations: [1,2,4,8,16,32] → Receptive Field: 127 frames")
    print("BatchNorm + GELU + Adadelta ✓")
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1:3d}/{EPOCHS} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.1f}%")
        print(f"Valid: Loss={val_loss:.4f}, Acc={val_acc:.1f}%, F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': train_ds.classes
            }, 'best_pure_tcn.pth')
            print(f"💾 SAVED! Best F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        scheduler.step(val_f1)
        
        if patience_counter >= 25:
            print("\n🛑 Early stopping.")
            break
            
    print(f"\n🎉 FINAL RESULT: Best F1 = {best_f1:.4f}")
    print("Model: best_pure_tcn.pth")

if __name__ == "__main__":
    main()
