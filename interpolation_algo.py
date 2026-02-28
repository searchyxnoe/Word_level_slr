# Interpolation algorithm for missing hands because zeroing harms some models, especially for small datasets :
# Written by Mistral AI model using devstral-latest-2512

import numpy as np
from typing import List, Dict, Optional

def preprocess_video_frames(raw_frames: List[Dict[str, Optional[List[List[float]]]]]) -> np.ndarray:
    """
    Process raw video frames into a normalized 15x144 tensor for neural network input.

    Args:
        raw_frames: List of dictionaries containing pose and hand joint data for each frame.
                   Each frame should have keys: 'pose', 'left_hand', 'right_hand' with values
                   being lists of [x,y,z] coordinates or None if not detected.

    Returns:
        Normalized 15x144 numpy array ready for neural network input (float32).
    """
    # Step 1: Data Aggregation and Alignment
    if not raw_frames:
        return np.zeros((15, 144), dtype=np.float32)

    total_frames = len(raw_frames)
    # Initialize matrix with NaN (48 joints * 3 coordinates) - enforce float32
    joint_matrix = np.full((total_frames, 48, 3), np.nan, dtype=np.float32)

    for frame_idx, frame_data in enumerate(raw_frames):
        # Pose joints (6 joints) - enforce float32
        if frame_data['pose'] is not None:
            joint_matrix[frame_idx, :6, :] = np.array(frame_data['pose'], dtype=np.float32)

        # Left hand joints (21 joints) - enforce float32
        if frame_data['left_hand'] is not None:
            joint_matrix[frame_idx, 6:27, :] = np.array(frame_data['left_hand'], dtype=np.float32)

        # Right hand joints (21 joints) - enforce float32
        if frame_data['right_hand'] is not None:
            joint_matrix[frame_idx, 27:48, :] = np.array(frame_data['right_hand'], dtype=np.float32)

    # Step 2: Missing Data Imputation (1D Linear Interpolation)
    for joint in range(48):
        for coord in range(3):
            series = joint_matrix[:, joint, coord]
            nan_mask = np.isnan(series)

            # Only interpolate if we have some valid data but not all missing
            if nan_mask.any() and not nan_mask.all():
                all_idx = np.arange(total_frames)
                # np.interp automatically handles edge cases with flat extrapolation
                series[nan_mask] = np.interp(all_idx[nan_mask], all_idx[~nan_mask], series[~nan_mask])

    # Step 3: Temporal Downsampling
    frame_indices = np.linspace(0, total_frames - 1, num=15, dtype=np.int32)
    downsampled = joint_matrix[frame_indices]

    # Step 4: Anchor-Based Spatial Normalization (CRITICAL FIXES APPLIED)
    normalized_frames = np.zeros_like(downsampled)

    for frame_idx in range(15):
        frame = downsampled[frame_idx]

        # Normalize Pose (first 6 joints)
        pose_joints = frame[:6]
        if not np.all(np.isnan(pose_joints)):
            left_shoulder = pose_joints[0]
            right_shoulder = pose_joints[1]
            root_center = (left_shoulder + right_shoulder) * 0.5
            # CRITICAL FIX: Used max() to prevent exploding gradients from tracking glitches
            scale_factor = max(np.linalg.norm(left_shoulder - right_shoulder), 0.01)
            normalized_frames[frame_idx, :6] = (pose_joints - root_center) / scale_factor

        # Normalize Left Hand (joints 6-26)
        left_hand = frame[6:27]
        if not np.all(np.isnan(left_hand)):
            wrist = left_hand[0]
            middle_finger_base = left_hand[9]  # MediaPipe convention
            # CRITICAL FIX: Used max() limit
            scale_factor = max(np.linalg.norm(wrist - middle_finger_base), 0.01)
            normalized_frames[frame_idx, 6:27] = (left_hand - wrist) / scale_factor

        # Normalize Right Hand (joints 27-47)
        right_hand = frame[27:48]
        if not np.all(np.isnan(right_hand)):
            wrist = right_hand[0]
            middle_finger_base = right_hand[9]  # MediaPipe convention
            # CRITICAL FIX: Used max() limit
            scale_factor = max(np.linalg.norm(wrist - middle_finger_base), 0.01)
            normalized_frames[frame_idx, 27:48] = (right_hand - wrist) / scale_factor

    # Step 5: Feature Flattening and Fallback
    flattened = normalized_frames.reshape(15, -1)
    # Replace NaN with 0.0 and enforce float32
    return np.nan_to_num(flattened, nan=0.0).astype(np.float32)

# Example usage demonstrating all edge cases
if __name__ == "__main__":
    # Test case 1: Normal two-handed gesture
    normal_frame = {
        'pose': [
            [0.1, 0.2, 0.3],  # L_Shoulder
            [0.4, 0.5, 0.6],  # R_Shoulder
            [0.7, 0.8, 0.9],  # L_Elbow
            [1.0, 1.1, 1.2],  # R_Elbow
            [1.3, 1.4, 1.5],  # L_Wrist
            [1.6, 1.7, 1.8]   # R_Wrist
        ],
        'left_hand': [[2.0 + i*0.1, 2.1 + i*0.1, 2.2 + i*0.1] for i in range(21)],
        'right_hand': [[3.0 + i*0.1, 3.1 + i*0.1, 3.2 + i*0.1] for i in range(21)]
    }

    # Test case 2: One-handed gesture (right hand missing)
    one_hand_frame = {
        'pose': [
            [0.1, 0.2, 0.3],  # L_Shoulder
            [0.4, 0.5, 0.6],  # R_Shoulder
            [0.7, 0.8, 0.9],  # L_Elbow
            [1.0, 1.1, 1.2],  # R_Elbow
            [1.3, 1.4, 1.5],  # L_Wrist
            [1.6, 1.7, 1.8]   # R_Wrist
        ],
        'left_hand': [[2.0 + i*0.1, 2.1 + i*0.1, 2.2 + i*0.1] for i in range(21)],
        'right_hand': None  # Right hand completely missing
    }

    # Test case 3: Tracking glitch (hand collapsed to single point)
    glitch_frame = {
        'pose': [
            [0.1, 0.2, 0.3],  # L_Shoulder
            [0.4, 0.5, 0.6],  # R_Shoulder
            [0.7, 0.8, 0.9],  # L_Elbow
            [1.0, 1.1, 1.2],  # R_Elbow
            [1.3, 1.4, 1.5],  # L_Wrist
            [1.6, 1.7, 1.8]   # R_Wrist
        ],
        'left_hand': [[2.0, 2.1, 2.2]] * 21,  # All points collapsed to same location
        'right_hand': [[3.0 + i*0.1, 3.1 + i*0.1, 3.2 + i*0.1] for i in range(21)]
    }

    # Test all cases
    test_cases = [
        ("Normal two-handed", [normal_frame] * 30),
        ("One-handed gesture", [one_hand_frame] * 30),
        ("Tracking glitch", [glitch_frame] * 30)
    ]

    for name, frames in test_cases:
        result = preprocess_video_frames(frames)
        print(f"\n{name}:")
        print(f"  Shape: {result.shape}")
        print(f"  Data type: {result.dtype}")
        print(f"  Min value: {result.min():.6f}")
        print(f"  Max value: {result.max():.6f}")
        print(f"  Mean value: {result.mean():.6f}")
        print(f"  Sample first frame (first 10 features): {result[0, :10]}")
