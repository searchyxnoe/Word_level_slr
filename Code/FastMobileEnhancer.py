#Credits to Gemini 3.1 Pro and Devstral-2512 
#Just given if someone need to reutilize it, it's optimized for cpu, run fast so that it do not slow users

import cv2
import numpy as np

class FastMobileEnhancer:
    def __init__(self):
        # 1. Initialize CLAHE once. 
        # Clip limit 1.5 is the safe zone.
        # Tile grid 4x4 (instead of 8x8) is much faster on mobile CPUs 
        # while still providing excellent local contrast for hand tracking.
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Memory & CPU optimized CLAHE for real-time mobile inference.
        Returns an RGB image ready directly for MediaPipe.
        """
        # OPTIMIZATION 1: Use YUV instead of LAB. 
        # Y is the luminance (brightness) channel. 
        # YUV conversion is mathematically lighter on ARM CPUs than LAB.
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # OPTIMIZATION 2: In-place memory operation.
        # Instead of splitting all 3 channels into new memory arrays (cv2.split),
        # we extract only the Y channel via slicing. This saves RAM allocations.
        y_channel = yuv_frame[:, :, 0]

        # Apply CLAHE directly to the Y channel, overwriting it in-place
        yuv_frame[:, :, 0] = self.clahe.apply(y_channel)

        # OPTIMIZATION 3: Convert directly to RGB for MediaPipe.
        # Standard pipeline is BGR -> YUV -> BGR -> RGB (2 conversions).
        # We skip the BGR middleman and go YUV -> RGB directly.
        rgb_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2RGB)
        
        return rgb_frame

# ==========================================
# HOW TO USE IN YOUR SCRIPT
# ==========================================
# 1. Initialize outside the loop
# enhancer = FastMobileEnhancer()
#
# 2. Inside your camera loop:
# ret, frame = cap.read()
# 
# # Apply the fast enhancement (returns RGB)
# rgb_frame = enhancer.apply(frame)
# 
# # Send directly to MediaPipe (MediaPipe requires RGB)
# results = mediapipe_hands.process(rgb_frame)
