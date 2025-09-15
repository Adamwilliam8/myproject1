### 在运行代码时要在主程序前加上(要不远程服务器256个cpu就满了)
import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["TF_NUM_INTRAOP_THREADS"] = "1" 
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["HWLOC_COMPONENTS"] = "-gl" 
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

import numpy as np
import pandas as pd
import highway_env
import gymnasium as gym
import random
from matplotlib import pyplot as plt
import stable_baselines3 as sb3
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import CallbackList
import os
from model import Model
import sys
import re
import yaml
import shutil


# 读取配置文件
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

previous_episode_reward = -1

# 注意！！！这行是为了确保第一次时有 奖励函数文件 去替换原先的
os.system("python reward_initial_API.py")

### 主循环
for gpt_response in range(OPENAI_CONFIG["cycle_number"]):
    rewards_dir = 'rewards'
    model_dirs = [d for d in os.listdir(rewards_dir)
                  if os.path.isfile(os.path.join(rewards_dir, d, 'reward_function.py'))]

    if not model_dirs:
        print('No candidate reward functions found')
        sys.exit(0)

    best_model = None
    best_reward = float('-inf')

    ## 循环2次，选出2个LLM产出的最好奖励函数
    for model_name in model_dirs:
        candidate_path = os.path.join(rewards_dir, model_name, 'reward_function.py')
        shutil.copy(candidate_path, 'reward_function.py') # 把 candidate_path 指向的源文件复制到当前工作目录下
        os.environ['REWARD_MODEL_NAME'] = model_name

        training_main_value = os.system('python train_main.py')
        if training_main_value != 0:
            print('error from train_main', training_main_value)
            sys.exit(0)
        testing_main_value = os.system('python test_main.py')
        if testing_main_value != 0:
            print('error from test_main', testing_main_value)
            sys.exit(0)

        with open('test_evaluations.txt', 'r') as file:
            content = file.read()
        match = re.search(r"总奖励 - 平均:\s*(-?\d+(?:\.\d+)?)", content)
        if match:
            mean_reward = float(match.group(1))
            print(f"{model_name} 测试平均奖励：", mean_reward)
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_model = model_name
        else:
            print("没有找到平均奖励结果")
    if best_model is None:
        print('No valid rewards found')
        sys.exit(0)
    # 把得到最好测试结果model的奖励函数 复制到当前工作目录下
    shutil.copy(os.path.join(rewards_dir, best_model, 'reward_function.py'),
                'reward_function.py')   

    ### 如果合格/不合格，怎么办
    if previous_episode_reward < 1.1 * best_reward:
        previous_episode_reward = best_reward
        print("analyzer API")
        analyzer_API_value = os.system("python analyzer_API.py")
        if analyzer_API_value != 0:
            print('error from analyzer_API_value', analyzer_API_value)
            sys.exit(0)

        print("reward modify API")
        reward_modify_API_value = os.system("python reward_modify_API.py")
        if reward_modify_API_value != 0:
            print('error from reward_modify_API_value', reward_modify_API_value)
            sys.exit(0)
    else:
        print("reward exploration API")
        reward_exploration_API_value = os.system("python reward_exploration_API.py")
        if reward_exploration_API_value != 0:
            print('error from reward_exploration_API_value', reward_exploration_API_value)
            sys.exit(0)
        

