# Data augmentation strategy applied to one of my datasets which I use, check readme to know more :
# Generated through Gemini 3.1 Pro, using an advice from Mistral AI model (advice vectorized matrix operations for cpu processing)

import os
import cv2
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==========================================================
# 1. CORE AUGMENTATION FUNCTIONS (Vectorized C++)
# ==========================================================
def rotate_image(image, angle):
    """Rotates an image by a specific angle using optimized affine transformations."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    # Get rotation matrix and apply it via OpenCV's highly optimized C++ backend
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return rotated_image

def count_mp4_files_fast(directory):
    """The fastest way to count files in Python using os.scandir (C-level OS call)."""
    count = 0
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.endswith(".mp4"):
            count += 1
    return count

def process_single_video(task_data):
    """
    Reads a single video and creates 5 augmented versions.
    Runs entirely within an isolated CPU core worker.
    """
    input_video_path, class_dir, starting_index = task_data
    
    # Define our target augmentations in a list to map to indices
    augmentations = [
        lambda frame: rotate_image(frame, 5),   # +5 degrees
        lambda frame: rotate_image(frame, 10),  # +10 degrees
        lambda frame: rotate_image(frame, -5),  # -5 degrees
        lambda frame: rotate_image(frame, -10), # -10 degrees
        lambda frame: cv2.flip(frame, 1)        # Horizontal flip
    ]

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return f"Error: Could not open {input_video_path}"

    # Get video properties to construct the output writers
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize VideoWriters. Names will be e.g., 9.mp4, 10.mp4, 11.mp4...
    writers = []
    for i in range(len(augmentations)):
        out_name = f"{starting_index + i}.mp4"
        out_path = os.path.join(class_dir, out_name)
        writers.append(cv2.VideoWriter(out_path, fourcc, fps, (width, height)))

    # Read frame-by-frame, augment, and write simultaneously
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply each augmentation and write to the respective output file
        for idx, aug_function in enumerate(augmentations):
            augmented_frame = aug_function(frame)
            writers[idx].write(augmented_frame)

    # Cleanup memory
    cap.release()
    for writer in writers:
        writer.release()

    return f"Success: Augmented {os.path.basename(input_video_path)} -> generated {starting_index}.mp4 to {starting_index + 4}.mp4"

# ==========================================================
# 2. BATCH PROCESSING MANAGER
# ==========================================================
def augment_dataset(root_dir):
    """Scans the directory and assigns video processing tasks to CPU workers."""
    tasks = []
    
    # Find all subdirectories (classes like 'alert', 'careful', etc.)
    class_dirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]

    for class_dir in class_dirs:
        # Fast-count existing mp4s (should be 8 initially)
        existing_count = count_mp4_files_fast(class_dir)
        
        # If we already have 40+ files, this class was already augmented
        if existing_count >= 40:
            print(f"Skipping {os.path.basename(class_dir)} - already fully augmented ({existing_count} files found).")
            continue
            
        # Get ONLY the original videos (files 1.mp4 to 8.mp4)
        # We assume the original files are numbered 1 to existing_count
        for i in range(1, existing_count + 1):
            video_path = os.path.join(class_dir, f"{i}.mp4")
            
            if os.path.exists(video_path):
                # Calculate what number the newly generated files should start at.
                # For video 1, it generates 9, 10, 11, 12, 13
                # For video 2, it generates 14, 15, 16, 17, 18
                # Formula: existing_count + ((i - 1) * 5) + 1
                starting_index = existing_count + ((i - 1) * 5) + 1
                
                tasks.append((video_path, class_dir, starting_index))

    total_tasks = len(tasks)
    print(f"Found {total_tasks} original videos to augment.")
    if total_tasks == 0:
        return

    # Use all CPU cores minus 1
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting isolated multiprocessing pool with {num_workers} CPU cores...")

    # Execute tasks in parallel across CPU cores
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_video, task) for task in tasks]

        for future in as_completed(futures):
            completed += 1
            result = future.result()
            print(f"[{completed}/{total_tasks}] {result}")

if __name__ == "__main__":
    # Point this to your training directory
    TRAINING_DIR = "/home/unknown_device/Musique/Hackathon/training/"
    
    print("Starting CPU-Optimized Data Augmentation...")
    augment_dataset(TRAINING_DIR)
    print("\nData Augmentation Complete!")
