# -*- coding: utf-8 -*-
# @Time    : 2024/6/29 16:20
# @Author  : Adam
# @File    : reward_initial.py
import gymnasium as gym
import os
import yaml

### 导入 配置文件config.yaml
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

reward_initial_prompt = '''
# REWARD INITIAL

You are now a proficient reward designer for a reinforcement learning (RL) agent. You need to write proper reward functions for the agent. The agent will be trained for the vehicle driving on highway driving environment to improve the performance of the agent. The detailed description and code of the task is as below.

    -Description: The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed, staying on the rightmost lanes and avoiding collisions. The agent objective is to reach a high speed while avoiding collisions with neighboring vehicles. Driving on the right side of the road is also rewarded. 
    -Environment code:
        {environement_code}
        {road}
        {controller}
        {kinematics}
        {action}
        {abstract}

## Reward function requirements

You should write a reward function to achieve the **Description**. The information you can use to formulate the reward function has been listed in the **Environment code**. 

## Output Requirements

- The reward function should be written in Python 3.11.1
- Output the code block only. **Do not output anything else outside the code block**.
- You should include **sufficient comments** in your reward function to explain your thoughts, the objective and **implementation details**. The implementation can be specified to a specific line of code.
- Ensure the reward value will not be extremely large or extremely small which makes the reward meaningless.
- If you need to import packages or define helper functions, MUST define functions or import packages in reward function.
- DO NOT use any undefined functions or variables or unimported packages in the reward function.
- Your reward function should use useful variables from the **Environment code** as inputs.
- Make sure code is compatible with TorchScript (e.g., **use torch tensor instead of numpy array**) since reward function will be decorated with @torch.jit.script.
- Make sure any new tensor or variable you introduce is on the same device as the input tensors.

## Output Helpful Tips

- Normalize the reward to a fixed range by applying transformations like torch.exp to the overall reward or its components is helpful.
- If you choose to transform a reward component, then you must also introduce a temperature parameter inside the transformation function; this parameter must be a named variable in the reward function and it must not be an input variable. Each transformed reward component should have its own temperature variable.
- Make sure the type of each input variable is correctly specified; a float input variable should not be specified as torch. Tensor. 
- Most importantly, the reward code\'s input variables must contain only attributes of the provided environment class definition (namely, variables that have prefix self.). Under no circumstance can you introduce new input variables. 

## Output Format

Strictly follow the following format. **Do not output anything else outside the code block**. **Do not use unimported packages and undefined functions**.

    def _reward(self, action) -> float:
        # Thoughts:
        # ...
        # (initial the reward)
        reward = 0.0
        # (import packages and define helper functions)
        ...
        return reward
'''

user_reward_initial="Now write a reward function. Then in each iteration, I will use the reward function to train the RL agent and test it in the environment. I will give you possible reasons for the failure found during the testing, and you should modify the reward function accordingly. **Do not output anything else outside the code block. Please double check the output code. Ensure there is no error. The variables or function used should be defined already.**"

### 填充提示词
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

reward_initial_prompt=reward_initial_prompt.format(environement_code=environement_code, 
                                                   road=road, 
                                                   controller=controller, 
                                                   kinematics=kinematics,
                                                   action=action,
                                                   abstract=abstract
                                                   )

with open('reward_initial_prompt.txt', 'w') as rip:
    print(reward_initial_prompt,file = rip)
    print(user_reward_initial,file=rip)