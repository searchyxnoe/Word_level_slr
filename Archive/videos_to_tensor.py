# Code generated with Gemini 3.1 Pro 
# It sample 15 equidistant frames, enhance frame quality with CLAHE (luminosity enhancement for mediapipe), 
# then it extract the keypoints/landmarks from the 15 frames using hand and pose models and use interpolation if 
# some point are missing, then normalization is applied, finally you get a [15,144] numpy tensor
import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import gc
import time

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATHS = {
    'pose': '/home/unknown_device/Musique/Hackathon/models/pose_landmarker_full.task',
    'hand': '/home/unknown_device/Musique/Hackathon/models/hand_landmarker.task'
}
OUTPUT_DIR = '/home/unknown_device/Musique/Hackathon/processed'
FRAME_COUNT = 15
BATCH_SIZE = 5  # Optimal for AMD MagicBook memory

POSE_INDICES = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13,    'right_elbow': 14,
    'left_wrist': 15,    'right_wrist': 16
}

class FastMobileEnhancer:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv_frame[:, :, 0] = self.clahe.apply(yuv_frame[:, :, 0])
        return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)

# ==========================================
# WORKER GLOBALS & LIFECYCLE MANAGEMENT
# ==========================================
worker_pose_detector = None
worker_hand_detector = None
worker_enhancer = None

def init_worker():
    global worker_pose_detector, worker_hand_detector, worker_enhancer
    try:
        worker_enhancer = FastMobileEnhancer()

        base_options_pose = python.BaseOptions(model_asset_path=MODEL_PATHS['pose'])
        worker_pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base_options_pose,
                output_segmentation_masks=False,
                running_mode=vision.RunningMode.IMAGE
            )
        )

        base_options_hand = python.BaseOptions(model_asset_path=MODEL_PATHS['hand'])
        worker_hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options_hand,
                num_hands=2,
                running_mode=vision.RunningMode.IMAGE
            )
        )
    except Exception as e:
        print(f"Worker init failed: {str(e)}")
        raise

def close_worker():
    global worker_pose_detector, worker_hand_detector
    try:
        if worker_pose_detector:
            worker_pose_detector.close()
            worker_pose_detector = None
        if worker_hand_detector:
            worker_hand_detector.close()
            worker_hand_detector = None
    except Exception as e:
        print(f"Worker close failed: {str(e)}")

# ==========================================
# EXTRACTION & SAFE MATH LOGIC
# ==========================================
def extract_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < FRAME_COUNT:
            return []

        frame_indices = np.linspace(0, total_frames-1, num=FRAME_COUNT, dtype=np.int32)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        return frames
    finally:
        cap.release()

def preprocess_video_frames(raw_frames: List[Dict[str, Optional[List[List[float]]]]]) -> np.ndarray:
    """Uses the mathematically safe, crash-proof logic for NaNs."""
    if not raw_frames:
        return np.zeros((FRAME_COUNT, 144), dtype=np.float32)

    joint_matrix = np.full((FRAME_COUNT, 48, 3), np.nan, dtype=np.float32)

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
                all_idx = np.arange(FRAME_COUNT)
                series[nan_mask] = np.interp(all_idx[nan_mask], all_idx[~nan_mask], series[~nan_mask])

    normalized_frames = np.zeros_like(joint_matrix)
    for frame_idx in range(FRAME_COUNT):
        frame = joint_matrix[frame_idx]

        pose_joints = frame[:6]
        if not np.all(np.isnan(pose_joints)):
            left_shoulder, right_shoulder = pose_joints[0], pose_joints[1]
            root_center = (left_shoulder + right_shoulder) * 0.5
            scale_factor = max(np.linalg.norm(left_shoulder - right_shoulder), 0.01)
            normalized_frames[frame_idx, :6] = (pose_joints - root_center) / scale_factor

        left_hand = frame[6:27]
        if not np.all(np.isnan(left_hand)):
            wrist, middle_finger_base = left_hand[0], left_hand[9]
            scale_factor = max(np.linalg.norm(wrist - middle_finger_base), 0.01)
            normalized_frames[frame_idx, 6:27] = (left_hand - wrist) / scale_factor

        right_hand = frame[27:48]
        if not np.all(np.isnan(right_hand)):
            wrist, middle_finger_base = right_hand[0], right_hand[9]
            scale_factor = max(np.linalg.norm(wrist - middle_finger_base), 0.01)
            normalized_frames[frame_idx, 27:48] = (right_hand - wrist) / scale_factor

    flattened = normalized_frames.reshape(FRAME_COUNT, -1)
    return np.nan_to_num(flattened, nan=0.0).astype(np.float32)

def process_video_task(args: Tuple[str, str, str]) -> str:
    video_path, split, class_name = args
    global worker_pose_detector, worker_hand_detector, worker_enhancer

    try:
        frames = extract_frames(video_path)
        if not frames:
            return f"Failed (No frames): {os.path.basename(video_path)}"

        raw_keypoints = []
        for frame in frames:
            enhanced_frame = worker_enhancer.apply(frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced_frame)

            pose_result = worker_pose_detector.detect(mp_image)
            pose_keypoints = None
            if pose_result.pose_landmarks:
                lms = pose_result.pose_landmarks[0]
                pose_keypoints = [
                    [lms[POSE_INDICES['left_shoulder']].x, lms[POSE_INDICES['left_shoulder']].y, lms[POSE_INDICES['left_shoulder']].z],
                    [lms[POSE_INDICES['right_shoulder']].x, lms[POSE_INDICES['right_shoulder']].y, lms[POSE_INDICES['right_shoulder']].z],
                    [lms[POSE_INDICES['left_elbow']].x, lms[POSE_INDICES['left_elbow']].y, lms[POSE_INDICES['left_elbow']].z],
                    [lms[POSE_INDICES['right_elbow']].x, lms[POSE_INDICES['right_elbow']].y, lms[POSE_INDICES['right_elbow']].z],
                    [lms[POSE_INDICES['left_wrist']].x, lms[POSE_INDICES['left_wrist']].y, lms[POSE_INDICES['left_wrist']].z],
                    [lms[POSE_INDICES['right_wrist']].x, lms[POSE_INDICES['right_wrist']].y, lms[POSE_INDICES['right_wrist']].z]
                ]

            hand_result = worker_hand_detector.detect(mp_image)
            left_hand = right_hand = None
            if hand_result.hand_landmarks:
                for i, hand_lms in enumerate(hand_result.hand_landmarks):
                    kpts = [[lm.x, lm.y, lm.z] for lm in hand_lms]
                    if hand_result.handedness[i][0].category_name == 'Right':
                        left_hand = kpts
                    else:
                        right_hand = kpts

            raw_keypoints.append({'pose': pose_keypoints, 'left_hand': left_hand, 'right_hand': right_hand})

        processed_tensor = preprocess_video_frames(raw_keypoints)

        output_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        output_path = os.path.join(output_class_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npy")
        np.save(output_path, processed_tensor)

        del frames, raw_keypoints, processed_tensor
        return f"Success: {split}/{class_name}/{os.path.basename(output_path)}"

    except Exception as e:
        return f"Error on {os.path.basename(video_path)}: {str(e)}"

# ==========================================
# ORCHESTRATION
# ==========================================
def process_dataset(dataset_root: str) -> None:
    video_paths = []

    for split in ['training', 'validation']:
        split_path = os.path.join(dataset_root, split)
        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for video_file in os.listdir(class_path):
                if video_file.endswith('.mp4'):
                    video_paths.append((os.path.join(class_path, video_file), split, class_name))

    total_tasks = len(video_paths)
    if total_tasks == 0:
        print("No videos found to process")
        return

    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting processing with {max_workers} workers...")
    print(f"Total videos to process: {total_tasks}")

    start_time = time.time()
    completed = 0
    success_count = 0

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            mp_context=multiprocessing.get_context('spawn')
        ) as executor:
            for i in range(0, total_tasks, BATCH_SIZE):
                batch = video_paths[i:i+BATCH_SIZE]
                for result in executor.map(process_video_task, batch):
                    completed += 1
                    if result.startswith("Success"):
                        success_count += 1
                    print(f"[{completed}/{total_tasks}] {result}")

                    if completed % 20 == 0:
                        gc.collect()

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Fatal error in processing: {str(e)}")
    finally:
        print("\nClosing worker processes...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(lambda _: close_worker(), range(max_workers))
            
        elapsed = time.time() - start_time
        print(f"Processing completed in {elapsed:.2f} seconds")
        print(f"Success rate: {success_count}/{total_tasks} ({100*success_count/total_tasks:.1f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/unknown_device/Musique/Hackathon')
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)
    process_dataset(args.dataset)
