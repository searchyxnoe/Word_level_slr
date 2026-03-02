# Context
Mistral AI Hackathon project (online track) : 
Building a deep learning model which learn spatio temporal features, of sign gestures, through mediapipe keypoints/landmarks, in order to translate signed gestures (static/dynamic) to text, so that developpers/ong can work upon, to allow the deaf community to interact with the society and receive appropriate service.

# Overall architecture / Plan
1. Find a dataset of RGB videos : https://www.kaggle.com/datasets/jahanzebnaeem/wlpsl Licensed under the Computational Use of Data Agreement (C-UDA) 
2. Improve frame quality using CLAHE on extracted frames only so that mediapipe recognition model is improved
3. Apply data augmentation to make dataset bigger
4. Extract hands landmarks and pose landmarks through mediapipe to obtain 3d world coordinate so that we can differentiate complex gestures
5. Fix missing value & apply normalization => store data in a [15,138] tensor
6. Train a ML or DL model for Validation_F1 >=90%
7. Check performance and iterate until i am pleased with the results (unfortunately I didnt organize myself well, neither reflected enough well)

# Constraints
1. Training locally in a laptop : 16 GB Ram, AMD 4600 H, no dedicated GPU => cpu-only training => explore light architectures
2. 2 days hackathon (I was already familiar to the subjects through technological watch)
3. Word level sign language recognition
4. Make it accessible to devices through efficient model which run in cpu only environments
5. Integrate a Mistral model to form sentences from recognized words according to SLR grammar rules
   
# Journaling 
1. 01/03/2026 - 13h20 : I restart extracting data according to another strategy due to the poor performance for real time testing :(
                        Thus an archive folder will be created to showcase previous work of saturday
2. 02/03/2026 - 00:54 : Updating github repo to make the research accessible, archive folder is considered useless for me but I prefer to let you see
                        I am almost pleased with new data extraction strategy, but the fact that we are using less data from pose model may be a cue
                        for the performance ceiling (val f1 = 90%)
3.

