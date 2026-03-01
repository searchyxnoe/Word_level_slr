Class_name_i.npy = [Normalized_Pose_Joints (Shoulders, Elbows, Wrists),Normalized_Left_Hand_Fingers, Normalized_Right_Hand_Fingers]

A. Body Position (Where are the hands?)

    Take Shoulders (L, R), Elbows (L, R), and Wrists (L, R) from the Pose model.

    Center them at the Root (mid-point of shoulders).

    Divide by Shoulder_Width.

    Total: 6 joints × 3 coords = 18 values.

B. Left Hand Shape (What is the left hand doing?)

    Take all 21 joints from the Left Hand model.

    Center them at the Left_Hand_Wrist (landmark 0).

    Divide by the Left_Hand_Size (distance from wrist to middle finger base).

    Total: 21 joints × 3 coords = 63 values.

C. Right Hand Shape (What is the right hand doing?)

    Take all 21 joints from the Right Hand model.

    Center them at the Right_Hand_Wrist (landmark 0).

    Divide by the Right_Hand_Size (distance from wrist to middle finger base).

    Total: 21 joints × 3 coords = 63 values.

Concatenate A + B + C to get Class_name_i.npy = [Normalized_Pose_Joints (Shoulders, Elbows, Wrists),Normalized_Left_Hand_Fingers, Normalized_Right_Hand_Fingers]
