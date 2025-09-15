# -*- coding: utf-8 -*-
"""
Created on 2024/6/29 16:20

@author: Adam
"""

import os
import yaml
from utils import load_truncated_trajectories
from utils import find_newest_model_dir

### 导入 配置文件config.yaml
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

analyzer1 = '''
You are now a proficient reward designer for a reinforcement learning (RL) agent. The agent will be trained for the vehicle driving on highway driving environment to improve the performance of the agent.. The detailed description of the task is in the following section. I now have a reward function for the agent to complete the described task. I have trained the RL agent for several times and tested it in the simulation environment. I will give you the information on the training and test results. You should help me write a proper analysis of possible reasons for inefficiency from both the training and test performances and your suggestions on the reward function improvement. 

## Task description and code

- Description: The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed, staying on the rightmost lanes and avoiding collisions. The agent objective is to reach a high speed while avoiding collisions with neighboring vehicles. Driving on the right side of the road is also rewarded. 
- Environment code: 
        {environement_code}     
        {road}
        {controller}
        {kinematics}
        {action}
        {abstract}

## Current reward function

The reward function part in **environment_code** is **invalid** at the beginning of training. Use the following reward function instead:
  
{reward_function}

## Input format

### Training evaluation

The first part is train_evaluations which contains the training evaluation results.  

The result are shown as several array keys, including:

- timesteps: An array of the number of training steps at each evaluation
- results: The average reward of each evaluation episode per evaluation
- ep_lengths: The average length of each evaluation episode per evaluation

### Testing evaluation

The second part is test_evaluations which contains the testing evaluation results. 

The result are shown as two types of lines, including:

- Snapshot every 50 steps (sample format): Step 50: Total reward = 0.967, Speed = 30.0, Lane = 1, Collision = False
- End summary of each episode (sample format): Episode 1 ends: Reward = 29.510, Training time expired
- The following info dictionary line contains info={'speed':..., 'crashed':..., 'action': array(...), 'rewards': {...}} (rewards includes components such as collision_reward, right_lane_reward, high_speed_reward, on_road_reward).

### Training and test trajectories

The third and forth part are training and test trajectories, which record part of model trajectories during training and testing.

The trajectories are shown as a list, each line is a list of an entire episode, and each element in the list is a dictionary of step. When episode has too many steps, it will be truncated to the last 50 steps.

The format is:
[
  {"obs":[...], "action":[0], "reward":0.12, "done":false, "truncated":false},
  {"obs":[...], "action":[1], "reward":-0.05, "done":false, "truncated":false},
  ...
  {"obs":[...], "action":[0], "reward":0.20, "done":true, "truncated":false}
]

where "obs" is current observation (ndarray → list), "action" is action (scalar or array → list/scalar), "reward" is reward (floating point), "done" is whether the episode is done (boolean), "truncated" is whether the episode is truncated due to a timeout (boolean).

## Output Requirements

Please write a proper analysis of the training performance (i.e., the convergence) and the test performance. Please try to give some suggestions on the reward improvement. You should not not be limited to the task description above, but also come up with other inefficient cases based on the training and test results.

**The analysis and suggestions should be concise.**
'''
analyzer2 = '''
## Training and test results

- Train Results:
{train_evaluations}

- Test Results:
{test_evaluations}

- Training Trajectories:
{train_trajectories}

- Test Trajectories:
{test_trajectories}

Now according to the **Training and test results**, please write your analysis and suggestions on the reward function improvement.
'''

### analyzer1填充提示词
file=open(os.path.join(OPENAI_CONFIG["ENV_FILE_ADDRESS"] , "envs" , "highway_env.py"), 'r', encoding='utf-8')
# file=open(os.path.join(os.getcwd(),"..","HighwayEnv-RL_env","highway_env","envs","highway_env.py"), 'r', encoding='utf-8')
content=file.readlines()
environement_code=''
for line in content:
    environement_code=environement_code+"    "+line
file.close()

file=open(os.path.join(OPENAI_CONFIG["ENV_FILE_ADDRESS"] ,"road","road.py"), 'r', encoding='utf-8')
# file=open(os.path.join(os.getcwd(),"..","HighwayEnv-RL_env","highway_env","road","road.py"), 'r', encoding='utf-8')
content=file.readlines()
road=''
for line in content:
    road=road+"    "+line
file.close()

file=open(os.path.join(OPENAI_CONFIG["ENV_FILE_ADDRESS"],"vehicle","controller.py"), 'r', encoding='utf-8')
content=file.readlines()
controller=''
for line in content:
    controller=controller+"    "+line
file.close()

file=open(os.path.join(OPENAI_CONFIG["ENV_FILE_ADDRESS"],"vehicle","kinematics.py"), 'r', encoding='utf-8')
content=file.readlines()
kinematics=''
for line in content:
    kinematics=kinematics+"    "+line
file.close()

file=open(os.path.join(OPENAI_CONFIG["ENV_FILE_ADDRESS"],"envs","common","action.py"), 'r', encoding='utf-8')
content=file.readlines()
action=''
for line in content:
    action=action+"    "+line
file.close()

file=open(os.path.join(OPENAI_CONFIG["ENV_FILE_ADDRESS"],"envs","common","abstract.py"), 'r', encoding='utf-8')
content=file.readlines()
abstract=''
for line in content:
    abstract=abstract+"    "+line
file.close()

file=open(os.path.join(os.getcwd(),"reward_function.py"), 'r', encoding='utf-8')
content=file.readlines()
reward_function=''
for line in content:
    reward_function=reward_function+"    "+line
file.close()

### analyzer2填充提示词
with open("train_evaluations.txt", "r", encoding="utf-8") as file:
    train_evaluations = file.read()

with open("test_evaluations.txt", "r", encoding="utf-8") as file:
    test_evaluations = file.read()

train_trajectories = load_truncated_trajectories("train_trajectories.jsonl")
test_trajectories = load_truncated_trajectories("test_trajectories.jsonl")

analyzer_all = analyzer1.format(environement_code=environement_code, road=road, controller=controller, kinematics=kinematics,action=action,abstract=abstract,reward_function=reward_function)
user = analyzer2.format(
    train_evaluations=train_evaluations,
    test_evaluations=test_evaluations,
    train_trajectories=train_trajectories,
    test_trajectories=test_trajectories,
)

data_path = find_newest_model_dir(OPENAI_CONFIG["tensorboard_log_path"])

with open(os.path.join(data_path,'analyzer_prompt.txt'), 'w') as ap:
    print(analyzer_all,file = ap)
    print(user,file=ap)