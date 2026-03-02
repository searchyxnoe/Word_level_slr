#Writed by Mistral
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp

# Model classes (same as before)
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

# Feature extraction functions (same as before)
class FastMobileEnhancer:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))

    def apply(self, frame):
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv_frame[:, :, 0]
        yuv_frame[:, :, 0] = self.clahe.apply(y_channel)
        return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)

def compute_bones(landmarks, wrist_idx=0, scale_idx=12):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    wrist = pts[wrist_idx]
    pts_relative = pts - wrist
    hand_length = np.linalg.norm(pts_relative[scale_idx])
    if hand_length > 1e-6:
        pts_norm = pts_relative / hand_length
    else:
        pts_norm = pts_relative

    bones = [
        (0,1), (1,2), (2,3), (3,4),
        (0,5), (5,6), (6,7), (7,8),
        (0,9), (9,10), (10,11), (11,12),
        (0,13), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20)
    ]

    bone_vectors = []
    for parent, child in bones:
        bone_vec = pts_norm[child] - pts_norm[parent]
        bone_vectors.extend(bone_vec)
    return np.array(bone_vectors, dtype=np.float32)

def extract_features_from_frames(frames):
    enhancer = FastMobileEnhancer()
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    )
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5
    )

    raw_features = []
    for frame in frames:
        rgb_frame = enhancer.apply(frame)
        hands_res = mp_hands.process(rgb_frame)
        pose_res = mp_pose.process(rgb_frame)

        left_bones = np.zeros(60, dtype=np.float32)
        right_bones = np.zeros(60, dtype=np.float32)
        left_present, right_present = 0.0, 0.0

        if hands_res.multi_hand_world_landmarks and hands_res.multi_handedness:
            for hand_landmarks, handedness in zip(hands_res.multi_hand_world_landmarks, hands_res.multi_handedness):
                label = handedness.classification[0].label
                if label == "Left":
                    left_bones = compute_bones(hand_landmarks)
                    left_present = 1.0
                elif label == "Right":
                    right_bones = compute_bones(hand_landmarks)
                    right_present = 1.0

        left_arm_pos = np.zeros(3, dtype=np.float32)
        right_arm_pos = np.zeros(3, dtype=np.float32)

        if pose_res.pose_world_landmarks:
            plm = pose_res.pose_world_landmarks.landmark
            l_shoulder = np.array([plm[11].x, plm[11].y, plm[11].z])
            r_shoulder = np.array([plm[12].x, plm[12].y, plm[12].z])
            midpoint = (l_shoulder + r_shoulder) / 2.0

            l_wrist = np.array([plm[15].x, plm[15].y, plm[15].z])
            r_wrist = np.array([plm[16].x, plm[16].y, plm[16].z])

            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
            if shoulder_width > 1e-6:
                left_arm_pos = (l_wrist - midpoint) / shoulder_width
                right_arm_pos = (r_wrist - midpoint) / shoulder_width

        if left_present == 0.0 and right_present == 0.0:
            raw_features.append(None)
        else:
            f_vec = np.concatenate([
                left_bones, right_bones,
                left_arm_pos, right_arm_pos,
                [left_present, right_present]
            ])
            raw_features.append(f_vec)

    interp_features = []
    for i in range(len(raw_features)):
        if raw_features[i] is not None:
            interp_features.append(raw_features[i])
        else:
            search_radius = 3
            prev_valid, next_valid = -1, -1

            for step in range(1, search_radius + 1):
                if i - step >= 0 and raw_features[i - step] is not None and prev_valid == -1:
                    prev_valid = i - step
                if i + step < len(raw_features) and raw_features[i + step] is not None and next_valid == -1:
                    next_valid = i + step

            if prev_valid != -1 and next_valid != -1:
                alpha = (i - prev_valid) / (next_valid - prev_valid)
                f_interp = (1 - alpha) * raw_features[prev_valid] + alpha * raw_features[next_valid]
                f_interp[-2:] = 0.0
                interp_features.append(f_interp)
            elif prev_valid != -1:
                f_fill = raw_features[prev_valid].copy()
                f_fill[-2:] = 0.0
                interp_features.append(f_fill)
            elif next_valid != -1:
                f_fill = raw_features[next_valid].copy()
                f_fill[-2:] = 0.0
                interp_features.append(f_fill)
            else:
                interp_features.append(np.zeros(128, dtype=np.float32))

    final_tensor = np.zeros((len(interp_features), 134), dtype=np.float32)
    for i in range(len(interp_features)):
        final_tensor[i, :120] = interp_features[i][:120]
        final_tensor[i, 120:126] = interp_features[i][120:126]

        if i == 0:
            final_tensor[i, 126:132] = 0.0
        else:
            prev_arm_pos = interp_features[i-1][120:126]
            curr_arm_pos = interp_features[i][120:126]
            final_tensor[i, 126:132] = curr_arm_pos - prev_arm_pos

        final_tensor[i, 132:134] = interp_features[i][126:128]

    mp_hands.close()
    mp_pose.close()
    return final_tensor

# Video recording and processing with confidence threshold
class VideoRecorder:
    def __init__(self, confidence_threshold=0.7):
        self.recording = False
        self.frames = []
        self.cap = cv2.VideoCapture(0)
        self.window_name = "Sign Language Recognition"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("best_pure_tcn.pth", map_location=self.device)
        self.classes = checkpoint['classes']
        self.model = PureTCN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # ROI variables
        self.roi = None
        self.selecting_roi = False
        self.roi_start = None
        self.confidence_threshold = confidence_threshold

        # Mouse callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting_roi = True
            self.roi_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting_roi = False
            x, y = x, y
            if self.roi_start:
                x1, y1 = self.roi_start
                self.roi = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))

    def start_recording(self):
        self.recording = True
        self.frames = []
        print("Recording started...")

    def stop_recording(self):
        self.recording = False
        print("Recording stopped. Processing...")
        if len(self.frames) > 0:
            # Extract 15 equidistant frames
            total_frames = len(self.frames)
            indices = np.linspace(0, total_frames - 1, 15, dtype=int)
            selected_frames = [self.frames[i] for i in indices]

            # Extract features
            features = extract_features_from_frames(selected_frames)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Predict with confidence
            with torch.no_grad():
                output = self.model(features_tensor)
                probs = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                confidence = confidence.item()
                predicted_class = self.classes[predicted.item()]

            if confidence > self.confidence_threshold:
                print(f"Predicted class: {predicted_class} (Confidence: {confidence:.2f})")
                cv2.putText(self.frames[-1], f"Predicted: {predicted_class} ({confidence:.2f})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                print(f"Low confidence prediction: {predicted_class} (Confidence: {confidence:.2f})")
                cv2.putText(self.frames[-1], "Low confidence", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(self.window_name, self.frames[-1])
            cv2.waitKey(2000)

    def run(self):
        print(f"Press 'r' to start/stop recording. Draw ROI with mouse. Confidence threshold: {self.confidence_threshold}")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Draw ROI if selected
            if self.roi:
                x1, y1, x2, y2 = self.roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if self.recording:
                    # Crop to ROI
                    roi_frame = frame[y1:y2, x1:x2]
                    # Resize to maintain aspect ratio
                    h, w = roi_frame.shape[:2]
                    if h > w:
                        new_h = 480
                        new_w = int(w * (480 / h))
                    else:
                        new_w = 640
                        new_h = int(h * (640 / w))
                    roi_frame = cv2.resize(roi_frame, (new_w, new_h))
                    # Pad to 640x480
                    if new_w < 640:
                        pad = (640 - new_w) // 2
                        roi_frame = cv2.copyMakeBorder(roi_frame, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
                    if new_h < 480:
                        pad = (480 - new_h) // 2
                        roi_frame = cv2.copyMakeBorder(roi_frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
                    self.frames.append(roi_frame)
            else:
                if self.recording:
                    # Resize to 640x480
                    frame = cv2.resize(frame, (640, 480))
                    self.frames.append(frame)

            # Display instructions
            status = "Recording" if self.recording else "Ready"
            cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'r' to toggle recording", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Draw ROI with mouse", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recorder = VideoRecorder(confidence_threshold=0.7)
    recorder.run()
