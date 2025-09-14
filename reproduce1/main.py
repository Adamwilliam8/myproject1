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

### 在运行代码时要在主程序前加上(要不远程服务器256个cpu就满了)
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["TF_NUM_INTRAOP_THREADS"] = "1" 
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["HWLOC_COMPONENTS"] = "-gl" 
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

# 读取配置文件
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

previous_episode_reward = -1

# 注意！！！这行是为了确保第一次时有 奖励函数文件 去替换原先的
os.system("python reward_initial_API.py")

env = gym.make("highway-v0", max_episode_steps=1000)
obs, info=env.reset() 


### 主循环
for gpt_response in range(OPENAI_CONFIG["cycle_number"]):  # 6~10
    # run train and test
    training_main_value = os.system("python train_main.py")  # 退出码
    if training_main_value != 0:
        print('error from train_main',training_main_value)
        sys.exit(0)
    testing_main_value = os.system("python test_main.py")    # 退出码
    if testing_main_value != 0:
        print('error from test_main',testing_main_value)
        sys.exit(0)

    ### 提取测试结果中的平均奖励
    file=open('test_evaluations.txt','r')
    content=file.read()
    match = re.search(r"总奖励 - 平均:\s*(-?\d+(?:\.\d+)?)", content)
    if match:
        mean_reward = float(match.group(1))
        print("测试平均奖励：", mean_reward)
    else:
        print("没有找到平均奖励结果")
    file.close()

    ## 如果合格/不合格，怎么办
    if previous_episode_reward < 1.1 * mean_reward:
        previous_episode_reward = mean_reward
        # run gpt    
        print("analyzer API")
        analyzer_API_value = os.system("python analyzer_API.py")
        if analyzer_API_value != 0:
            print('error from analyzer_API_value',analyzer_API_value)
            sys.exit(0)
        
        print("reward modify API")
        reward_modify_API_value = os.system("python reward_modify_API.py")
        if reward_modify_API_value != 0:
            print('error from reward_modify_API_value',reward_modify_API_value)
            sys.exit(0)
    else:
        print("reward exploration API")
        reward_exploration_API_value = os.system("python reward_exploration_API.py")
        if reward_exploration_API_value != 0:
            print('error from reward_exploration_API_value',reward_exploration_API_value)
            sys.exit(0)
