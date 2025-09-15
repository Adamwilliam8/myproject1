# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:26:52 2024

@author: Adam
"""

#import analyzer_API
import os
import yaml
from utils import find_newest_model_dir
from utils import load_truncated_trajectories

### 导入 配置文件config.yaml
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

exploration1= '''
You are now a proficient reward designer for a reinforcement learning (RL) agent. The agent will be trained for the vehicle driving on highway driving environment to improve the performance of the agent. The detailed description of the task is in the following section. 
Based on the previous reward function and the analysis as well as suggestions on the previous reward improvements, there is a modified reward function. The RL agent has been trained with the modified reward function, but the test results are not as good as those trained with the previous reward function. I will provide you with the modified reward function and corresponding test results as a failure modification. Please modify the previous reward function again based on the analysis and suggestions on the previous reward function, and considering the failure modification.

## Task description and code

- Description: The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed, staying on the rightmost lanes and avoiding collisions. The agent objective is to reach a high speed while avoiding collisions with neighboring vehicles. Driving on the right side of the road is also rewarded. 
- Environment code: 
        {environement_code}
        {road}
        {controller}
        {kinematics}
        {action}
        {abstract}

## Reward function requirements

Please modify the previous reward function based on **Analysis and suggestions on previous reward function**, considering the **Failure modification**. The reward function needs to achieve the **Description**. The information you can use to formulate the reward function has been listed in the **Environment code**. 

## Previous reward function

{previous_reward_function}

## Analysis and suggestions on previous reward function  

The previous reward function is used to train the reinforcement learning agent several times. Here is some analysis of failure and inefficiency and suggestions on the previous reward function:

{analysis}

## Failure modification

This is a failure modification of the previous reward function. The test results of the agent trained with the modified reward function are not as good as those trained with the previous reward function. This failure modification should be avoided when you modify the previous reward function again.

### Modified reward function
  
{reward_function}

### Train evaluation results of modified reward function

This part contains the training evaluation results.  

The result are shown as several array keys, including:

- timesteps: An array of the number of training steps at each evaluation
- results: The average reward of each evaluation episode per evaluation
- ep_lengths: The average length of each evaluation episode per evaluation

The following is Train Results:
{train_evaluations}

### Test evaluation results of modified reward function

This part contains the testing evaluation results. 

The result are shown as two types of lines, including:

- Snapshot every 50 steps (sample format): Step 50: Total reward = 0.967, Speed = 30.0, Lane = 1, Collision = False
- End summary of each episode (sample format): Episode 1 ends: Reward = 29.510, Training time expired
- The following info dictionary line contains info={'speed':..., 'crashed':..., 'action': array(...), 'rewards': {...}} (rewards includes components such as collision_reward, right_lane_reward, high_speed_reward, on_road_reward).

The following is Train Results:
{test_evaluations}

### Train and Test Trajectories of modified reward function

This part contains part of model trajectories during training and testing.

The trajectories are shown as a list, each line is a list of an entire episode, and each element in the list is a dictionary of step. When episode has too many steps, it will be truncated to the last 50 steps.

The format is:
[
  {"obs":[...], "action":[0], "reward":0.12, "done":false, "truncated":false},
  {"obs":[...], "action":[1], "reward":-0.05, "done":false, "truncated":false},
  ...
  {"obs":[...], "action":[0], "reward":0.20, "done":true, "truncated":false}
]

where "obs" is current observation (ndarray → list), "action" is action (scalar or array → list/scalar), "reward" is reward (floating point), "done" is whether the episode is done (boolean), "truncated" is whether the episode is truncated due to a timeout (boolean).

The following is Training Trajectories:
{train_trajectories}

The following is Test Trajectories:
{test_trajectories}

## Output Requirements

- Please consider the analysis and suggestions above. Modify and improve the current reward function. 
    1. You can both modify the current lines and add new lines. You can use any variable in the **Environment code** to define the reward function.  
    2. If necessary, you can write a **totally different** reward function than the current one.
    3. Consider modifying the reward and penalty values in the current reward function to balance them.
    4. In the first part of the reward function, you should provide your thoughts on modifying the reward function. **The thoughts should be concise.**
    5. Ensure the reward value will not be extremely large or extremely small which makes the reward meaningless.
- The reward function should be written in Python 3.11.1
- Output the code block only. **Do not output anything else outside the code block**.
- You should include **sufficient comments** in your reward function to explain your thoughts, the objective and **implementation details**. The implementation can be specified to a specific line of code.
- If you need to import packages (e.g. math, numpy) or define helper functions, define them at the beginning of the function. Do not use unimported packages and undefined functions.
- Your reward function should use useful variables from the **Environment code** as inputs.
- Make sure code is compatible with TorchScript (e.g., use torch tensor instead of numpy array) since reward function will be decorated with @torch.jit.script
- Make sure any new tensor or variable you introduce is on the same device as the input tensors.
- **Please double check the output code. Ensure there is no error. The variables or function used should be defined already**

## Output Format

Strictly follow the following format. **Do not output anything else outside the code block**.

    def _reward(self, action) -> float:
        # Thoughts:
        # ...
        # (initial the reward)
        reward = 0.0
        # (import packages and define helper functions)
        ...
        return reward
'''

### env填充提示词
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

### 非env填充词
file = open('previous_reward_function.py','r')
previous_reward_function_lines = file.readlines()
previous_reward_function = ''
for item in previous_reward_function_lines:
    previous_reward_function = previous_reward_function + "    " + item
file.close()

file = open('previous_analyzer_output.txt','r')
analysis_lines = file.readlines()
analysis = ''
for item in analysis_lines:
    analysis = analysis + item
file.close()

file = open('reward_function.py','r')
reward_function_lines = file.readlines()
reward_function = ''
for item in reward_function_lines:
    reward_function = reward_function + "    " + item
file.close()

with open("train_evaluations.txt", "r", encoding="utf-8") as file:
    train_evaluations = file.read()

with open("test_evaluations.txt", "r", encoding="utf-8") as file:
    test_evaluations = file.read()

train_trajectories = load_truncated_trajectories("train_trajectories.jsonl")
test_trajectories = load_truncated_trajectories("test_trajectories.jsonl")

reward_exploration_final=exploration1.format(environement_code=environement_code,
                                              road=road,
                                              controller=controller,
                                              kinematics=kinematics,
                                              action=action,
                                              abstract=abstract,
                                              previous_reward_function=previous_reward_function,
                                              analysis=analysis,
                                              reward_function=reward_function,
                                              train_evaluations=train_evaluations,
                                              test_evaluations=test_evaluations,
                                              train_trajectories=train_trajectories,
                                              test_trajectories=test_trajectories,
                                              )

user_reward_exploration = "Now write a new reward function to improve the **Previous reward function** based on **Analysis and suggestions on previous reward function**, and considering **Failure modification**. I will use the new reward function to train the RL agent and test it in the environment. **Do not output anything else outside the code block**. **Please double-check the output code. Ensure there is no error. The variables or functions used should be defined already.**" 

data_path = find_newest_model_dir(OPENAI_CONFIG["tensorboard_log_path"])

with open(data_path + 'reward_exploration_prompt.txt', 'w') as rmp:
    print(reward_exploration_final,file = rmp)
    print(user_reward_exploration,file=rmp)
    
with open(data_path + 'previous_reward_function.py','w') as write_prev_reward1:
    #print('# -*- coding: utf-8 -*-')
    print(previous_reward_function,file=write_prev_reward1)