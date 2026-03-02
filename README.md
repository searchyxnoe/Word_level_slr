# Context
Mistral AI Hackathon project (online track) : 
Building a deep learning model which learn spatio temporal features, of sign gestures, through mediapipe keypoints/landmarks, in order to translate signed gestures (static/dynamic) to text, so that developpers/ong can work upon, to allow the deaf community to interact with the society and receive appropriate service.

# Prototype 
Link YTB : https://www.youtube.com/watch?v=-8j1cZY1qAs

# Contact
If you want to contact me to help me (non profit project) : 
discord : alphanord 
email : xavier-shiller@proton.me 

# Overall architecture / Plan (We do not recognize gestures using raw videos, thus training is not on videos but on extracted fingers and some relevant data)
1. Find a dataset of RGB videos : https://www.kaggle.com/datasets/jahanzebnaeem/wlpsl Licensed under the Computational Use of Data Agreement (C-UDA) 
2. Improve frame quality using CLAHE on extracted frames only so that mediapipe recognition model is improved
3. Apply data augmentation to make dataset bigger
4. Extract hands landmarks and pose landmarks through mediapipe to obtain 3d world coordinate so that we can differentiate complex gestures
5. Fix missing value & apply normalization => store data in a [15,138] tensor
6. Train a ML or DL model for Validation_F1 >=90%
7. Implement an NPL model for sentence printing from recognized words using a small Mistral model (unfortunately time is running)
8. Check performance and iterate until i am pleased with the results (unfortunately I didnt organize myself well, neither reflected enough well)
   Spoiler : I'm happy I advanced and learned so much due to restrained time, choice of dataset and poor data augmentation did not allow
   to achieve the ambition of val f1 > 95%, but I believe feasability of a true real time slr system using TCN for multi class classification
   or retrieval due to prior experiences with it. 
9. Deploy it for cross platform (actually only ui mockup made by Qwen 3.5 Plus, it generated some html files then Gemini 3.1 Pro refined)

# Detailed Architecture implemented 
Input: (B, 15, 134)
  ↓ transpose → (B, 134, 15)
  ↓ Conv1D(134→64, k=1)  # Projection
  ↓ TCNBlock(ch=64, d=1)
  ↓ TCNBlock(ch=64, d=2)
  ↓ TCNBlock(ch=64, d=4)
  ↓ TCNBlock(ch=64, d=8)
  ↓ TCNBlock(ch=64, d=16)
  ↓ TCNBlock(ch=64, d=32)  # Receptive field = 127
  ↓ AdaptiveAvgPool1D(1) → (B, 64)
  ↓ Linear(64→32) → BN → GELU → Drop(0.3)
  ↓ Linear(32→31 classes)

Each TCNBlock = 2x Dilated Conv + Residual (repeat 6x with growing dilation)

# Requirements 

Python 3.11 , Pytorch, UV (I use  for managing python versions and libraries)
For libraries versions look at https://github.com/searchyxnoe/Word_level_slr/blob/main/pyproject.toml 

Or 

To make testing easier I'm zipping the folder I used in my pc, for this Hackathon so people can test the program locally, easily, just adapt the path folder for Live_Inference.py and have a camera and enjoy, download here : https://kdrive.infomaniak.com/app/share/269184/a29c9ccc-5059-48fc-b514-b40369729577  (kdrive is a swiss alternative to gmail btw)

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
3. 02/03/2026 - 03:58 : Start submitting the work in Hackiterate video link : https://www.youtube.com/watch?v=-8j1cZY1qAs

