# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:32:26 2024

@author: Adam
"""

#import analyzer_API
import os
import yaml
from utils import find_newest_model_dir

### 导入 配置文件config.yaml
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

modify1= '''
You are now a proficient reward designer for a reinforcement learning (RL) agent. The agent will be trained for the vehicle driving on highway driving environment to improve the performance of the agent. The detailed description of the task is in the following section. I now have a reward function. The reward function has been used to train the RL agent several times and is tested in the environment. I will provide you with the current reward function, an analysis on the performance of the current reward function, and suggestions for reward improvement. You should help me modify and improve the current reward function.

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

You should write a reward function to achieve the **Description**. The information you can use to formulate the reward function has been listed in the **Environment code**. 

## Current reward function

The reward function part in **environment_code** is **invalid** at the beginning of training. Use the following reward function instead:
  
{reward_function}

The reward function is used to train the reinforcement learning agent several times. Here is some analysis of the agent's performance and suggestions for the current reward function improvement:

{analysis}

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

data_path = find_newest_model_dir(OPENAI_CONFIG["tensorboard_log_path"])

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
file = open('reward_function.py','r')
reward_function_lines = file.readlines()
reward_function = ''
for item in reward_function_lines:
    reward_function = reward_function + "    " + item
file.close()

file = open(os.path.join(data_path,"analyzer_output.txt"),'r')
analysis_lines = file.readlines()
analysis = ''
for item in analysis_lines:
    analysis = analysis + item
file.close()

reward_modify_final=modify1.format(environement_code=environement_code, road=road, controller=controller, kinematics=kinematics,action=action,abstract=abstract,reward_function=reward_function,analysis=analysis)

user_reward_modify = "Now write a new reward function to improve the current one based on **Analysis and suggestions for current reward function**. I will use the new reward function to train the RL agent and test it in the environment. **Do not output anything else outside the code block**. **Please double check the output code. Ensure there is no error. The variables or functions used should be defined already.**" 

with open(data_path + 'reward_modify_prompt.txt', 'w') as rmp:
    print(reward_modify_final,file = rmp)
    print(user_reward_modify,file=rmp)
    
with open('previous_reward_function.py','w') as write_prev_reward:
    #print('# -*- coding: utf-8 -*-')
    print(reward_function,file=write_prev_reward)

with open('previous_analyzer_output.txt','w') as write_prev_analyze:
    print(analysis, file=write_prev_analyze)
    
with open(data_path + 'previous_reward_function.py','w') as write_prev_reward1:
    #print('# -*- coding: utf-8 -*-')
    print(reward_function,file=write_prev_reward1)