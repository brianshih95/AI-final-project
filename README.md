# AI-final-project

## Introduction:<br />
This repository implements reinforcement learning to automatically detect landmarks in 3D medical images.

## prerequisite:<br />
If you want to run DQN:<br />
cd DQN<br />
pip install -r requirements.txt

If you want to run Q-learning:<br />
First, run cd Q-learning and pip install -r requirements.txt.<br />
Then, since our environment is derived from the gym's taxi environment, you will need to copy the content of environment.py to :<br />
(virtual environment name)\Lib\site-packages\gym\envs\toy_text\taxi.py

## Running the code:<br />
If you want to run DQN:<br />
cd src<br />
python DQN.py --task eval --load 'data/models/BrainMRI/SingleAgent.pt' --files 'data/filenames/image_files.txt' 'data/filenames/landmark_files.txt' --file_type brain --landmarks 13 --model_name "Network3d"

If you want to run Q-learning:<br />
python main.py

## Result:<br />
Q-learning:<br />
![image](https://github.com/brianshih95/AI-final-project/blob/main/Q-learning/result/4.gif)
<br />