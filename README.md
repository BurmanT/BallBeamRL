# BallBeamRL

Create conda environment on terminal: **conda create -n myenv python=3.8.10**

Activate conda environment: **conda activate myenv**

Clone repo and cd: **pip install -e .**

Run: **bash script.sh**

script.sh: runs qlearning.py with different reward functions and discretizations of observation space 

qlearning.py: script to run a QLearning agent 

main.py: script to run PG model for a certain number of runs, saves models, and figures

ball_beam/envs script for RL gym environment 

ball_beam/resources has urdf files and action class for the pybullet 
