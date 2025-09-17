import os
### 在运行代码时要在主程序前加上(要不远程服务器256个cpu就满了)
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
from utils import copy_directory, copy_file


# 读取配置文件
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

previous_episode_reward = -1

# 注意！！！这行是为了确保第一次时有 描述环境的文件
os.system("python env_analyzer_API.py")
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

    ### 创造temp文件夹，复制现在的情况到基线
    main_model_dir = os.path.abspath(OPENAI_CONFIG["tensorboard_log_path"])
    temp_root = os.path.abspath('temp')
    os.makedirs(main_model_dir, exist_ok=True)
    os.makedirs(temp_root, exist_ok=True)

    baseline_dir = os.path.join(temp_root, 'baseline_model')
    if os.path.exists(baseline_dir):
        shutil.rmtree(baseline_dir)
    copy_directory(main_model_dir, baseline_dir)


    best_model = None
    best_reward = float('-inf')
    best_candidate_info = None
    candidate_records = []

    ## 循环2次，选出2个LLM产出的最好奖励函数
    for idx, model_name in enumerate(model_dirs):
        candidate_path = os.path.join(rewards_dir, model_name, 'reward_function.py')
        shutil.copy(candidate_path, 'reward_function.py') # 把 candidate_path 指向的源文件复制到当前工作目录下
        os.environ['REWARD_MODEL_NAME'] = model_name

        ### 复制基线到 candidate_xxx，准备传给train/test, 写入东西
        candidate_dir = os.path.abspath(os.path.join(temp_root, f'candidate_{idx}'))
        if os.path.exists(candidate_dir):
            shutil.rmtree(candidate_dir)
        copy_directory(baseline_dir, candidate_dir)
        os.environ['MODEL_WORKSPACE'] = candidate_dir

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
        else:
            print("没有找到平均奖励结果")
            mean_reward = float('-inf')

        candidate_info = {
            'index': idx,
            'name': model_name,
            'path': candidate_dir,
            'mean_reward': mean_reward,
        }

        dqn_dirs = [
            d for d in os.listdir(candidate_dir)
            if os.path.isdir(os.path.join(candidate_dir, d))
            and d.startswith('DQN_')
            and d.split('_')[1].isdigit()
        ]
        if dqn_dirs:
            latest_dqn = max(dqn_dirs, key=lambda d: int(d.split('_')[1]))
        else:
            latest_dqn = None
        candidate_info['latest_dqn_dir'] = latest_dqn
        candidate_records.append(candidate_info)

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model = model_name
            best_candidate_info = candidate_info

    os.environ['MODEL_WORKSPACE'] = main_model_dir

    if best_model is None or best_candidate_info is None:
        print('No valid rewards found')
        sys.exit(0)

    # 把得到最好测试结果model的奖励函数 复制到当前工作目录下
    shutil.copy(os.path.join(rewards_dir, best_model, 'reward_function.py'),
                'reward_function.py')   

    ### 知道哪个LLM奖励函数最好了，把各项数据搬运到主目录
    best_idx = best_candidate_info['index']
    best_candidate_dir = best_candidate_info['path']
    print(f"最佳候选索引: {best_idx}, 目录: {best_candidate_dir}")

    checkpoints_src = os.path.join(best_candidate_dir, 'checkpoints')
    if os.path.exists(checkpoints_src):
        copy_directory(checkpoints_src, os.path.join(main_model_dir, 'checkpoints'))

    best_eval_src = os.path.join(best_candidate_dir, 'best_Eval')
    if os.path.exists(best_eval_src):
        copy_directory(best_eval_src, os.path.join(main_model_dir, 'best_Eval'))

    copy_file(os.path.join(best_candidate_dir, 'final_model.zip'),
              os.path.join(main_model_dir, 'final_model.zip'))

    latest_dqn_dir = best_candidate_info.get('latest_dqn_dir')
    if latest_dqn_dir:
        copy_directory(os.path.join(best_candidate_dir, latest_dqn_dir),
                       os.path.join(main_model_dir, latest_dqn_dir))

    train_trajectories_src = os.path.join(best_candidate_dir, 'train_trajectories.jsonl')
    if os.path.exists(train_trajectories_src):
        shutil.copy(train_trajectories_src, 'train_trajectories.jsonl')

    test_trajectories_src = os.path.join(best_candidate_dir, 'test_trajectories.jsonl')
    if os.path.exists(test_trajectories_src):
        shutil.copy(test_trajectories_src, 'test_trajectories.jsonl')

    for record in candidate_records:
        if record['path'] != best_candidate_dir and os.path.exists(record['path']):
            shutil.rmtree(record['path'])

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
        

