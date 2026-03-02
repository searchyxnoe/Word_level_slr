import os
import cv2
import gc
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==========================================
# 1. ENHANCER & PIPELINE DEFINITIONS
# ==========================================

class FastMobileEnhancer:
    def __init__(self):
        # Optimized for mobile/laptop CPU: 4x4 tiles, clip 1.5
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv_frame[:, :, 0]
        yuv_frame[:, :, 0] = self.clahe.apply(y_channel)  # In-place
        return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)

def get_equidistant_frames(video_path: str, num_frames: int = 15):
    """Reads a video and mathematically extracts exactly `num_frames` equidistant frames."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            if len(frames) > 0: frames.append(frames[-1].copy())
            else: frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if len(frames) > 0 else np.zeros((480, 640, 3), dtype=np.uint8))
        
    return frames[:num_frames]

def compute_bones(landmarks, wrist_idx=0, scale_idx=12):
    """Wrist-centric translation, scale normalization, and 20 bone vectors."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    wrist = pts[wrist_idx]
    pts_relative = pts - wrist
    
    hand_length = np.linalg.norm(pts_relative[scale_idx])
    if hand_length > 1e-6:
        pts_norm = pts_relative / hand_length
    else:
        pts_norm = pts_relative
        
    bones = [
        (0,1), (1,2), (2,3), (3,4),       # Thumb
        (0,5), (5,6), (6,7), (7,8),       # Index
        (0,9), (9,10), (10,11), (11,12),  # Middle
        (0,13), (13,14), (14,15), (15,16),# Ring
        (0,17), (17,18), (18,19), (19,20) # Pinky
    ]
    
    bone_vectors = []
    for parent, child in bones:
        bone_vec = pts_norm[child] - pts_norm[parent]
        bone_vectors.extend(bone_vec)
        
    return np.array(bone_vectors, dtype=np.float32)

# ==========================================
# 2. CORE EXTRACTION LOGIC (PER VIDEO)
# ==========================================

def process_video_worker(args):
    """
    Self-contained worker that creates and destroys its own MediaPipe graph.
    Takes a tuple of args to map cleanly with ProcessPoolExecutor.
    """
    video_path, output_path = args
    
    # 1. Instantiate locally to avoid C++ state sharing across Linux processes
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    )
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5
    )
    enhancer = FastMobileEnhancer()
    
    try:
        frames = get_equidistant_frames(video_path, 15)
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
        for i in range(15):
            if raw_features[i] is not None:
                interp_features.append(raw_features[i])
            else:
                search_radius = 3
                prev_valid, next_valid = -1, -1
                
                for step in range(1, search_radius + 1):
                    if i - step >= 0 and raw_features[i - step] is not None and prev_valid == -1:
                        prev_valid = i - step
                    if i + step < 15 and raw_features[i + step] is not None and next_valid == -1:
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
                    
        final_tensor = np.zeros((15, 134), dtype=np.float32)
        
        for i in range(15):
            final_tensor[i, :120] = interp_features[i][:120]
            final_tensor[i, 120:126] = interp_features[i][120:126]
            
            if i == 0:
                final_tensor[i, 126:132] = 0.0
            else:
                prev_arm_pos = interp_features[i-1][120:126]
                curr_arm_pos = interp_features[i][120:126]
                final_tensor[i, 126:132] = curr_arm_pos - prev_arm_pos
                
            final_tensor[i, 132:134] = interp_features[i][126:128]

        np.save(output_path, final_tensor)
        return True
        
    except Exception as e:
        return f"Error on {video_path}: {str(e)}"
    
    finally:
        # 2. Hard close MediaPipe C++ bindings to free RAM immediately
        mp_hands.close()
        mp_pose.close()

# ==========================================
# 3. RAM-SAFE MULTIPROCESSING RUNNER
# ==========================================

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    base_dir = Path("/home/unknown_device/Musique/Hackathon/Videos")
    output_dir = Path("/home/unknown_device/Musique/Hackathon/npy_dataset")
    
    tasks = []
    
    for split in ["training", "validation"]:
        split_path = base_dir / split
        if not split_path.exists(): continue
            
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir(): continue
                
            out_class_dir = output_dir / split / class_dir.name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            for video_file in class_dir.glob("*.mp4"):
                out_file = out_class_dir / f"{video_file.stem}.npy"
                if not out_file.exists():
                    tasks.append((str(video_file), str(out_file)))

    print(f"Found {len(tasks)} videos to process.")
    if len(tasks) == 0:
        return

    # Use CPU cores safely
    cpu_cores = max(1, multiprocessing.cpu_count() - 1)
    
    # We break the 1400+ tasks into chunks of 50. 
    # The pool will process 50 videos, totally shut down to clear RAM, then start a new pool.
    chunk_size = 50 
    task_chunks = list(chunk_list(tasks, chunk_size))
    
    with tqdm(total=len(tasks), desc="Extracting Tensors (Batched MP)") as pbar:
        for chunk in task_chunks:
            # Create a fresh executor for each chunk to prevent long-term memory leaks
            with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
                futures = [executor.submit(process_video_worker, args) for args in chunk]
                
                for future in as_completed(futures):
                    res = future.result()
                    if res is not True:
                        print(res)
                    pbar.update(1)
            
            # Completely flush RAM after each chunk finishes
            gc.collect()

if __name__ == "__main__":
    # 'spawn' guarantees that no memory states are copied from the parent process
    multiprocessing.set_start_method("spawn", force=True)
    main()
