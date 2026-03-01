You can test model through data or live, in live you have to sign a sign once and wait that the window close to get prediction
Code made using Gemini 3.1 Pro 
---
Train + get per-class F1 on internal validation split:

bash
python your_script.py --mode train

    Recompute per-class F1 later without retraining:

bash
python your_script.py --mode eval

    Live webcam inference (record ~5s video, then classify):

bash
python your_script.py --mode live 
---
import os
import time
import gc
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = '/home/unknown_device/Musique/Hackathon/processed'
MODEL_PATHS = {
    'pose': '/home/unknown_device/Musique/Hackathon/models/pose_landmarker_full.task',
    'hand': '/home/unknown_device/Musique/Hackathon/models/hand_landmarker.task'
}
OUTPUT_MODEL = 'sign_language_slowfast_tcn.pth'

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
SEQ_LENGTH = 15
NUM_FEATURES = 144
NUM_CLASSES = 31   # keep in sync with your dataset
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = False

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==========================================
# DATASET
# ==========================================
class SignLanguageDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = data           # (N, 15, 144)
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def temporal_jitter(self, seq):
        if np.random.rand() < 0.5:
            shift = np.random.randint(-2, 3)
            seq = np.roll(seq, shift, axis=0)
        return seq

    def frame_dropout(self, seq):
        if np.random.rand() < 0.5:
            num_drop = np.random.randint(1, 3)
            idxs = np.random.choice(SEQ_LENGTH, size=num_drop, replace=False)
            seq[idxs] = 0.0
        return seq

    def joint_jitter(self, seq):
        if np.random.rand() < 0.8:
            noise = np.random.normal(loc=0.0, scale=0.01, size=seq.shape).astype(np.float32)
            seq = seq + noise
        return seq

    def __getitem__(self, idx):
        sample = self.data[idx].copy()   # (15, 144)
        if self.augment:
            sample = self.temporal_jitter(sample)
            sample = self.frame_dropout(sample)
            sample = self.joint_jitter(sample)
        sample = torch.from_numpy(sample).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


def load_data(data_dir):
    """
    Loads ALL .npy tensors from processed/training and processed/validation,
    uses integer class indices based on sorted training class_names.
    """
    X, y = [], []
    class_names = sorted(os.listdir(os.path.join(data_dir, 'training')))
    for split in ['training', 'validation']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(class_dir, file))
                    if data.shape == (SEQ_LENGTH, NUM_FEATURES):
                        X.append(data)
                        y.append(class_idx)
    return np.array(X), np.array(y), class_names

# ==========================================
# TCN BUILDING BLOCKS
# ==========================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNBackbone(nn.Module):
    def __init__(self, input_channels, num_levels, hidden_channels,
                 kernel_size=3, dropout=0.4):
        super().__init__()
        layers = []
        in_channels = input_channels
        for i in range(num_levels):
            dilation = 2 ** i
            out_channels = hidden_channels
            layers.append(
                TemporalBlock(in_channels, out_channels,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              dropout=dropout)
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        self.hidden_channels = hidden_channels

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()  # (B, F, T)
        y = self.tcn(x)                      # (B, H, T)
        y = y.mean(dim=2)                    # (B, H)
        return y

class SlowFastTCN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.fast_backbone = TCNBackbone(
            input_channels=input_channels,
            num_levels=3,
            hidden_channels=128,
            kernel_size=3,
            dropout=0.4
        )
        self.slow_backbone = TCNBackbone(
            input_channels=input_channels,
            num_levels=2,
            hidden_channels=64,
            kernel_size=3,
            dropout=0.4
        )
        fused_dim = 128 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.register_buffer(
            "slow_indices",
            torch.tensor([0, 3, 7, 11, 14], dtype=torch.long)
        )

    def forward(self, x):
        B, T, F = x.shape
        fast_feat = self.fast_backbone(x)
        idx = self.slow_indices
        x_slow = x.index_select(dim=1, index=idx)
        slow_feat = self.slow_backbone(x_slow)
        fused = torch.cat([fast_feat, slow_feat], dim=1)
        out = self.classifier(fused)
        return out

# ==========================================
# TRAINING + F1 EVALUATION
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Val)'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_acc)

        print(
            f'Epoch {epoch+1}/{num_epochs}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_slowfast_tcn_model.pth')
            print(f'*** New best SlowFast-TCN model saved with val acc: {val_acc:.2f}% ***')

    return train_losses, val_losses, train_accs, val_accs

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('slowfast_tcn_training_metrics.png')
    plt.show()

def evaluate_per_class(model, X_val, y_val, class_names, device):
    """
    Computes and prints per-class precision/recall/F1 on the internal validation split.
    """
    model.eval()
    dataset = SignLanguageDataset(X_val, y_val, augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    print("\n=== Per-class metrics on validation (internal split) ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1: {macro_f1:.4f}")

# ==========================================
# TRAIN ENTRYPOINT
# ==========================================
def train_main():
    print("Loading data...")
    X, y, class_names = load_data(DATA_DIR)

    # These y are already integer class indices; LabelEncoder is optional
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )

    print(f"Training samples: {X_train.shape[0]}, Validation: {X_val.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}, Num classes: {NUM_CLASSES}")

    train_loader = DataLoader(
        SignLanguageDataset(X_train, y_train, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        SignLanguageDataset(X_val, y_val, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SlowFastTCN(
        input_channels=NUM_FEATURES,
        num_classes=NUM_CLASSES
    ).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    print("Starting SlowFast-lite TCN training...")
    start_time = time.time()

    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler,
        device, EPOCHS
    )

    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    # Load best weights
    model.load_state_dict(torch.load('best_slowfast_tcn_model.pth', map_location=device))
    model.eval()

    # Save clean checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'input_size': NUM_FEATURES,
        'num_classes': NUM_CLASSES,
        'arch': 'SlowFastTCN_skeleton_fast15_slow5'
    }, OUTPUT_MODEL)
    print(f"SlowFast-lite TCN model saved as {OUTPUT_MODEL}")

    # Per-class F1 on internal validation split
    evaluate_per_class(model, X_val, y_val, class_names, device)

# ==========================================
# LIVE INFERENCE PIPELINE (single video)
# ==========================================
# We re-use your YUV CLAHE + interpolation + anchor normalization [file:298][file:297][web:278][web:279]

POSE_INDICES = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13,    'right_elbow': 14,
    'left_wrist': 15,    'right_wrist': 16
}

class FastMobileEnhancerLive:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    def apply(self, frame: np.ndarray) -> np.ndarray:
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv_frame[:, :, 0] = self.clahe.apply(yuv_frame[:, :, 0])
        return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)

class LiveVideoProcessor:
    def __init__(self):
        base_options_pose = python.BaseOptions(model_asset_path=MODEL_PATHS['pose'])
        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base_options_pose,
                output_segmentation_masks=False,
                running_mode=vision.RunningMode.IMAGE
            )
        )
        base_options_hand = python.BaseOptions(model_asset_path=MODEL_PATHS['hand'])
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options_hand,
                num_hands=2,
                running_mode=vision.RunningMode.IMAGE
            )
        )
        self.enhancer = FastMobileEnhancerLive()

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < SEQ_LENGTH:
            cap.release()
            return []
        frame_indices = np.linspace(0, total_frames-1, num=SEQ_LENGTH, dtype=np.int32)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    def preprocess_video_frames(self, raw_frames: List[Dict[str, Optional[List[List[float]]]]]) -> np.ndarray:
        if not raw_frames:
            return np.zeros((SEQ_LENGTH, NUM_FEATURES), dtype=np.float32)

        joint_matrix = np.full((SEQ_LENGTH, 48, 3), np.nan, dtype=np.float32)

        for frame_idx, frame_data in enumerate(raw_frames):
            if frame_data['pose'] is not None:
                joint_matrix[frame_idx, :6, :] = np.array(frame_data['pose'], dtype=np.float32)
            if frame_data['left_hand'] is not None:
                joint_matrix[frame_idx, 6:27, :] = np.array(frame_data['left_hand'], dtype=np.float32)
            if frame_data['right_hand'] is not None:
                joint_matrix[frame_idx, 27:48, :] = np.array(frame_data['right_hand'], dtype=np.float32)

        for joint in range(48):
            for coord in range(3):
                series = joint_matrix[:, joint, coord]
                nan_mask = np.isnan(series)
                if nan_mask.any() and not nan_mask.all():
                    all_idx = np.arange(SEQ_LENGTH)
                    series[nan_mask] = np.interp(all_idx[nan_mask], all_idx[~nan_mask], series[~nan_mask])

        normalized_frames = np.zeros_like(joint_matrix)
        for frame_idx in range(SEQ_LENGTH):
            frame = joint_matrix[frame_idx]

            pose_joints = frame[:6]
            if not np.all(np.isnan(pose_joints)):
                left_shoulder, right_shoulder = pose_joints[0], pose_joints[1]
                root_center = (left_shoulder + right_shoulder) * 0.5
                scale_factor = max(np.linalg.norm(left_shoulder - right_shoulder), 0.01)
                normalized_frames[frame_idx, :6] = (pose_joints - root_center) / scale_factor

            left_hand = frame[6:27]
            if not np.all(np.isnan(left_hand)):
                wrist, middle_finger = left_hand[0], left_hand[9]
                scale_factor = max(np.linalg.norm(wrist - middle_finger), 0.01)
                normalized_frames[frame_idx, 6:27] = (left_hand - wrist) / scale_factor

            right_hand = frame[27:48]
            if not np.all(np.isnan(right_hand)):
                wrist, middle_finger = right_hand[0], right_hand[9]
                scale_factor = max(np.linalg.norm(wrist - middle_finger), 0.01)
                normalized_frames[frame_idx, 27:48] = (right_hand - wrist) / scale_factor

        flattened = normalized_frames.reshape(SEQ_LENGTH, -1)
        return np.nan_to_num(flattened, nan=0.0).astype(np.float32)

    def process_video_to_tensor(self, video_path: str) -> np.ndarray:
        frames = self.extract_frames(video_path)
        if len(frames) != SEQ_LENGTH:
            print("Not enough frames extracted from live video.")
            return None

        raw_keypoints = []
        for frame in frames:
            enhanced = self.enhancer.apply(frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced)

            pose_result = self.pose_detector.detect(mp_image)
            pose_keypoints = None
            if pose_result.pose_landmarks:
                lms = pose_result.pose_landmarks[0]
                pose_keypoints = [
                    [lms[POSE_INDICES['left_shoulder']].x,  lms[POSE_INDICES['left_shoulder']].y,  lms[POSE_INDICES['left_shoulder']].z],
                    [lms[POSE_INDICES['right_shoulder']].x, lms[POSE_INDICES['right_shoulder']].y, lms[POSE_INDICES['right_shoulder']].z],
                    [lms[POSE_INDICES['left_elbow']].x,     lms[POSE_INDICES['left_elbow']].y,     lms[POSE_INDICES['left_elbow']].z],
                    [lms[POSE_INDICES['right_elbow']].x,    lms[POSE_INDICES['right_elbow']].y,    lms[POSE_INDICES['right_elbow']].z],
                    [lms[POSE_INDICES['left_wrist']].x,     lms[POSE_INDICES['left_wrist']].y,     lms[POSE_INDICES['left_wrist']].z],
                    [lms[POSE_INDICES['right_wrist']].x,    lms[POSE_INDICES['right_wrist']].y,    lms[POSE_INDICES['right_wrist']].z]
                ]

            hand_result = self.hand_detector.detect(mp_image)
            left_hand = right_hand = None
            if hand_result.hand_landmarks:
                for i, hand_lms in enumerate(hand_result.hand_landmarks):
                    kpts = [[lm.x, lm.y, lm.z] for lm in hand_lms]
                    if hand_result.handedness[i][0].category_name == 'Right':
                        left_hand = kpts
                    else:
                        right_hand = kpts

            raw_keypoints.append({'pose': pose_keypoints,
                                  'left_hand': left_hand,
                                  'right_hand': right_hand})

        tensor = self.preprocess_video_frames(raw_keypoints)
        return tensor  # (15, 144)

def record_video_from_camera(output_path: str, fps: int = 30, max_seconds: int = 5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Recording... press 'q' to stop earlier.")
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording (press q to stop)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start > max_seconds:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved live video to {output_path}")
    return True

def live_inference_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(OUTPUT_MODEL):
        print(f"Model checkpoint {OUTPUT_MODEL} not found. Train first.")
        return

    checkpoint = torch.load(OUTPUT_MODEL, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']

    model = SlowFastTCN(input_channels=NUM_FEATURES, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Loaded trained SlowFastTCN model for live inference.")

    temp_video = '/tmp/live_sign.mp4'
    if not record_video_from_camera(temp_video, fps=30, max_seconds=5):
        return

    processor = LiveVideoProcessor()
    features = processor.process_video_to_tensor(temp_video)
    if features is None:
        print("Could not extract valid tensor from live video.")
        return

    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0).float().to(device)  # (1, 15, 144)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        pred_idx = pred.item()
        conf_val = conf.item()
        print(f"\nPredicted class: {class_names[pred_idx]}  (confidence: {conf_val*100:.1f}%)")

# ==========================================
# EVAL-ONLY ENTRYPOINT (optional)
# ==========================================
def eval_main():
    """
    Reloads full dataset and best model; re-splits with same seed;
    recomputes per-class F1 on the internal validation split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(OUTPUT_MODEL):
        print(f"Model checkpoint {OUTPUT_MODEL} not found. Train first.")
        return

    X, y, class_names = load_data(DATA_DIR)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    _, X_val, _, y_val = train_test_split(
        X, y_encoded,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )

    checkpoint = torch.load(OUTPUT_MODEL, map_location=device)
    model = SlowFastTCN(input_channels=NUM_FEATURES,
                        num_classes=checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    evaluate_per_class(model, X_val, y_val, class_names, device)

# ==========================================
# MAIN DISPATCH
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'live'],
                        help='train: train + internal F1, eval: F1 only, live: webcam inference')
    args = parser.parse_args()

    if args.mode == 'train':
        train_main()
    elif args.mode == 'eval':
        eval_main()
    elif args.mode == 'live':
        live_inference_main()
