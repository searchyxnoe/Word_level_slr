# Context
Mistral AI Hackathon project (online track) : 
I am building a deep learning model to learn spatio temporal features, of sign gestures, through mediapipe keypoints/landmarks, in order to translate signed gestures (static/dynamic) to text, so that the deaf community can interact with the society.

# Overall architecture / Plan
1. Find a dataset of RGB videos (i have already somes)
2. Improve frame quality (luminosity CLAHE ?)
3. Apply data augmentation
4. Extract hands landmarks and pose landmarks through mediapipe
5. Fix missing value & apply normalization => store data in a [15,144] tensor
6. Train a ML or DL model
7. Check performance and iterate until i am pleased with the results
