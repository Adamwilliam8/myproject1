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


## Output REQUIREMENTS (must follow exactly):
- Output a JSON object only (no extra commentary). The JSON must contain these keys:
  1) "environment_overview": one-sentence summary of mission and core constraints.
  2) "default_config": dict of key config items and their default values (e.g. lanes_count, max_episode_steps, obs_type, action_type).
  3) "observations": list of {{ "index_or_key": ..., "name": ..., "type": "float|int|array|dict", "meaning": ..., "shape_hint": ... }} for each observation element used by reward/termination.
  4) "actions": {{ "type": "discrete|continuous", "mapping": {{index: meaning,...}}, "num": int }}
  5) "reward_components": list of components; each component is {{ "name": ..., "formula_summary": short text, "trigger": short text, "value_range": [min,max] or "unknown", "weight_key": config key or literal, "computed_in": "filename:func:approx_line" }}.
  6) "normalization": how reward is normalized/clipped (method and keys).
  7) "termination_truncation": list of conditions for done and truncated (collision, offroad, time limit, etc).
  8) "tunable_hooks": list of config keys relevant to reward tuning (names only).
  9) "code_references": list of minimal pointers where important logic lives: {{ "file": "...", "func": "...", "line_hint": "start-end or 'around X'" }}.
 10) "open_questions": list of missing info that affects reward design (e.g., unclear obs layout, unspecified action mapping). If none, put ["None"].

FORMAT RULES:
- Return exactly one valid JSON object (no prose outside).
- Keep string values short (<= 400 chars). If content was truncated, append "...TRUNCATED...".
- Where you quote a formula, keep it symbolic and short (e.g., "collision_reward: -1 if crashed else 0").
- Provide line hints (not full code) such as "highway_env/envs/common/abstract.py:_reward around line 240".
- If you cannot infer a value, use "unknown" (do not guess wildly).

PRIORITY:
- First, extract reward-related items and termination conditions.
- Second, extract observation elements that reward uses (speed, lane_index, distance_to_vehicle, etc).
- Third, list config keys and code references.
- CRITICAL DATA STRUCTURE WARNINGS:
    - Note that lane_index is usually a TUPLE (road_id, lane_id, longitudinal_position), NOT a scalar. Highlight this explicitly.
    - Any arithmetic with lane_index MUST extract the correct element first (typically lane_index[2] for actual lane number).
    - Show examples of correct lane_index access: `lane = self.vehicle.lane_index[2]  # Extract actual lane number`
    - Highlight the data type of each field in observations, especially when arrays/tuples vs scalar.
    - Document the possible error: "TypeError: unsupported operand type(s) for /: 'tuple' and 'int'" when using lane_index directly.
    - Make sure keep and highlight this warning in the "open_questions" section examples 
        - "open_questions":["Is lane_index a tuple or scalar? ANSWER: lane_index IS A TUPLE (road_id, lane_id, longitudinal_position)","Action->index mapping not explicit in action.py"]

TRUNCATION / SIZE:
- Do NOT include full source text. If a file is long, include only file path and a short line hint.

EXAMPLE OUTPUT:
{{
  "environment_overview": "Highway driving: maximize speed, stay rightmost lane, avoid collisions.",
  "default_config": {{"lanes_count": 4, "max_episode_steps": 500, "normalize_reward": true}},
  "observations": [{{"index_or_key": 0, "name": "ego_vehicle_state", "type": "array", "meaning": "position,speed,lane_index", "shape_hint": [10]}}],
  "actions": {{"type":"discrete","mapping":{{"0":"left","1":"keep","2":"right","3":"faster"}},"num":4}},
  "reward_components": [{{"name":"collision_reward","formula_summary":"-10 on crash","trigger":"crash in info","value_range":[-10,0],"weight_key":"collision_reward","computed_in":".../abstract.py:_reward around line 240"}}],
  "normalization":"clip to [-100,100] then scale by 0.01",
  "termination_truncation":["collision","offroad","time_limit"],
  "tunable_hooks":["right_lane_reward","high_speed_reward","collision_reward"],
  "code_references":[{{"file":"highway_env/envs/common/abstract.py","func":"_reward","line_hint":"~240"}}],
  "open_questions":["Is lane_index a tuple or scalar?","Action->index mapping not explicit in action.py"]
}}
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