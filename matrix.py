import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
from collections import defaultdict

# ================================
# 1. MODEL DEFINITION (SAME AS TRAINING)
# ================================
class TCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = dilation * 1
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)
        out = self.dropout(out)
        return out + residual

class PureTCN(nn.Module):
    def __init__(self, num_classes=31, input_dim=134, channels=64):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, channels, 1)
        self.tcn_layers = nn.ModuleList([
            TCNBlock(channels, dilation=2**i) for i in range(6)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.BatchNorm1d(channels // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(channels // 2, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for layer in self.tcn_layers:
            x = layer(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

# ================================
# 2. VALIDATION DATA LOADER
# ================================
class ValidationDataset:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir) / "validation"
        self.classes = sorted([d.name for d in self.base_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.data = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = self.base_dir / cls_name
            for npy_file in cls_dir.glob("*.npy"):
                self.data.append(str(npy_file))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.load(self.data[idx]).astype(np.float32)
        return torch.FloatTensor(data), torch.tensor(self.labels[idx], dtype=torch.long)

# ================================
# 3. EVALUATION FUNCTION
# ================================
def evaluate_model(model, data_loader, device, classes):
    model.eval()
    all_preds = []
    all_targets = []
    class_preds = defaultdict(list)
    class_targets = defaultdict(list)

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            for pred, target in zip(preds.cpu().numpy(), targets.cpu().numpy()):
                class_preds[classes[target]].append(pred)
                class_targets[classes[target]].append(target)

    # Overall metrics
    overall_f1 = f1_score(all_targets, all_preds, average='weighted')
    overall_acc = np.mean(np.array(all_preds) == np.array(all_targets))

    # Per-class metrics
    class_metrics = {}
    for cls_name in classes:
        if cls_name in class_preds:
            cls_f1 = f1_score(class_targets[cls_name], class_preds[cls_name], average='weighted')
            cls_acc = np.mean(np.array(class_preds[cls_name]) == np.array(class_targets[cls_name]))
            class_metrics[cls_name] = {
                'f1': cls_f1,
                'accuracy': cls_acc,
                'support': len(class_preds[cls_name])
            }

    return overall_f1, overall_acc, class_metrics

# ================================
# 4. MAIN EVALUATION
# ================================
def main():
    # Configuration
    BASE_DIR = "/home/unknown_device/Musique/Hackathon/npy_dataset"
    BATCH_SIZE = 8  # Keep small for validation
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("Loading validation data...")
    val_dataset = ValidationDataset(BASE_DIR)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    # Load model
    print("Loading model...")
    checkpoint = torch.load("best_pure_tcn.pth", map_location=DEVICE)
    classes = checkpoint['classes']
    model = PureTCN(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    print("Evaluating model...")
    overall_f1, overall_acc, class_metrics = evaluate_model(
        model, val_loader, DEVICE, val_dataset.classes
    )

    # Print results
    print("\n=== Overall Performance ===")
    print(f"Accuracy: {overall_acc:.4f}")
    print(f"Weighted F1 Score: {overall_f1:.4f}")

    print("\n=== Per-Class Performance ===")
    print("{:<15} {:<10} {:<10} {:<10}".format("Class", "F1 Score", "Accuracy", "Support"))
    for cls_name in sorted(class_metrics.keys()):
        metrics = class_metrics[cls_name]
        print("{:<15} {:<10.4f} {:<10.4f} {:<10}".format(
            cls_name,
            metrics['f1'],
            metrics['accuracy'],
            metrics['support']
        ))

    # Save results
    with open("validation_results.txt", "w") as f:
        f.write("=== Overall Performance ===\n")
        f.write(f"Accuracy: {overall_acc:.4f}\n")
        f.write(f"Weighted F1 Score: {overall_f1:.4f}\n\n")

        f.write("=== Per-Class Performance ===\n")
        f.write("{:<15} {:<10} {:<10} {:<10}\n".format("Class", "F1 Score", "Accuracy", "Support"))
        for cls_name in sorted(class_metrics.keys()):
            metrics = class_metrics[cls_name]
            f.write("{:<15} {:<10.4f} {:<10.4f} {:<10}\n".format(
                cls_name,
                metrics['f1'],
                metrics['accuracy'],
                metrics['support']
            ))

    print("\nResults saved to validation_results.txt")

if __name__ == "__main__":
    main()
