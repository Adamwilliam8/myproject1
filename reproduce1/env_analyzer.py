# -*- coding: utf-8 -*-
"""
Created on 2024/9/17 18:06

@author: Adam
"""

import os
import yaml

### 导入 配置文件config.yaml
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

env_analyzer='''
# ENVIRONMENT ANALYZER

You are now a proficient reinforcement learning environment analyst and agent reward designer. You know how to read the given Python and configuration files, analyze the environment, and extract key information about the reward function for downstream policy design. This agent will be trained on a highway driving environment to improve its performance. The task is described in detail below.

## Task description

You will now describe the HighwayEnv environment and its associated modules as detailed as possible in the following code file
    - {environement_code}     
    - {road}
    - {controller}
    - {kinematics}
    - {action}
    - {abstract}
, outputting a structured summary to help subsequent models understand how to design or improve rewards.

## Output Requirements

1. Focus on the HighwayEnv environment's objectives, core mission settings, and default configuration (number of lanes, number of vehicles, duration, etc.).
2. Describe the reward components, explaining their trigger conditions, value ranges, weights, and default values (e.g., collision_reward, right_lane_reward, high_speed_reward, on_road_reward, normalize_reward, reward_speed_range, etc.), as well as the normalization method and their linear combination in `_reward`.
3. Identify termination and truncation conditions (e.g., collision, leaving the track, time limit).
4. If the context includes variants (e.g., HighwayEnvFast or additional penalty/reward components), explain how these variants differ from the default environment and their design intent.
5. List key adjustable configurations: observation/action type, frequency, vehicle type, etc., and summarize the parameters directly related to reward design. 
6. List any unspecified information that might affect the reward design (e.g., missing weights, additional safety reward implementation locations) in an "Open Questions" section, explicitly marked as "Not used in current reward implementation."

**Please do not output code files directly, only use natural language description**

## Output format

**Strictly follow the following format**

- ## Environment Overview: Describe the mission objectives and core constraints in one sentence.
- ## Default Config Summary: List key configuration items and their default values in a table or bullet points.
- ## Reward Structure: Describe each reward item's name, formula, trigger conditions, value range, weight meaning and default value, and normalization/clipping logic.
- ## Termination and Truncation Conditions: Describe the episode termination conditions (e.g., the trigger conditions for `done` and `truncated`).
- ## TunableHooks: List configuration keys related to reward tunability.
- ## OpenQuestions: Missing information or suggested context; if none, write "None"

**Please strictly follow the above format and output in sections. Do not merge or omit any content.**
'''

user_env_analyzer='''Now extract the key information related to the reward function. Then in each iteration, I will provide the extracted information to other LLMs to generate or improve the reward function.'''

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

env_analyzer=env_analyzer.format(environement_code=environement_code, 
                                road=road, 
                                controller=controller, 
                                kinematics=kinematics,
                               action=action,
                               abstract=abstract
                                )

with open('env_analyzer_prompt.txt', 'w') as rip:
    print(env_analyzer,file = rip)
    print(user_env_analyzer,file=rip)