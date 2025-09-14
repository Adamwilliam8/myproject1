from model import Model
import highway_env
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import yaml
import glob
import re

### 读取配置文件
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
base_log_path = OPENAI_CONFIG["tensorboard_log_path"]  
os.makedirs(base_log_path, exist_ok=True)
existing_runs = [   # 获取 模型版本 最大数
    int(d.split("_")[1])
    for d in os.listdir(base_log_path)
    if os.path.isdir(os.path.join(base_log_path, d)) and d.startswith("DQN_") and d.split("_")[1].isdigit()
]
run_id = max(existing_runs) + 1 if existing_runs else 0
model_log_path = os.path.join(base_log_path, f"DQN_{run_id}")
os.makedirs(model_log_path, exist_ok=True)
model_kwargs = OPENAI_CONFIG["MODEL_KWARGS"]  ## 模型参数
model_kwargs["tensorboard_log"] = model_log_path

### 把新奖励函数绑定在环境上
env = gym.make("highway-v0")
obs, info=env.reset() 
# 把新写的奖励函数 绑定到当前的env实例上，覆盖掉原来类里的_reward方法
from reward_function import _reward
env._reward = _reward.__get__(env, type(env))  # 绑定为实例方法
# 评估环境（带 Monitor 且绑定自定义奖励）
eval_env = Monitor(gym.make("highway-v0"))
eval_env._reward = _reward.__get__(eval_env, type(eval_env))

### 查找最新的检查点
checkpoint_dir = os.path.join(OPENAI_CONFIG["tensorboard_log_path"], "checkpoints","")
if os.path.exists(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"加载检查点: {latest_checkpoint}")
    model=Model(env, model_class=getattr(sb3, OPENAI_CONFIG["MODEL_NAME"]), 
            model_kwargs=model_kwargs)
    model.load(latest_checkpoint,model_log_path) # 指定 model_log_path 为 tensorboard_log_path
    
    ## 创建一个新的日志记录器，设置到 model对象上
    new_logger = configure(model_log_path, ["stdout", "tensorboard"])
    model.model.set_logger(new_logger)

    model.train(total_timesteps=OPENAI_CONFIG["every_cycle_timesteps"], 
            eval_env=eval_env, 
            eval_freq=OPENAI_CONFIG["eval_freq"], 
            n_eval_episodes=OPENAI_CONFIG["n_eval_episodes"], 
            reward_threshold=OPENAI_CONFIG["reward_threshold"], 
            save_path=OPENAI_CONFIG["tensorboard_log_path"])
    model.save(os.path.join(OPENAI_CONFIG["tensorboard_log_path"], "final_model"))
else:
    print("没有找到检查点，创建新模型")
    model=Model(env, model_class=getattr(sb3, OPENAI_CONFIG["MODEL_NAME"]), 
            model_kwargs=model_kwargs)
    
    ## 创建一个新的日志记录器，设置到 model对象上
    new_logger = configure(model_log_path, ["stdout", "tensorboard"])
    model.model.set_logger(new_logger)
    
    model.train(total_timesteps=OPENAI_CONFIG["every_cycle_timesteps"], 
            eval_env=eval_env, 
            eval_freq=OPENAI_CONFIG["eval_freq"], 
            n_eval_episodes=OPENAI_CONFIG["n_eval_episodes"], 
            reward_threshold=OPENAI_CONFIG["reward_threshold"], 
            save_path=OPENAI_CONFIG["tensorboard_log_path"])
    model.save(os.path.join(OPENAI_CONFIG["tensorboard_log_path"], "final_model"))

print("------------------------------------------")
print("训练完成，模型已保存到 final_model.zip")
print(f"模型总训练步数：{model.num_timesteps}")
print("------------------------------------------")

### 创造文件 里面写入 训练的奖励函数(之后给ai分析用)
data = np.load(os.path.join(OPENAI_CONFIG["tensorboard_log_path"], 'best_Eval','evaluations.npz'))  
with open('train_evaluations.txt', 'w') as f:
    f.write("训练评估结果：\n\n")
    for key in data.keys():
        f.write(f"{key}:\n")
        array = data[key]
        if array.ndim == 1:
            # 一维数组，直接写入
            for value in array:
                f.write(f"{value}\n")
        elif array.ndim == 2:
            # 二维数组，逐行写入
            for row in array:
                f.write(" ".join(map(str, row)) + "\n")
        f.write("\n")  # 分隔不同数组

print("------------------------------------------")
print("转换完成，内容已写入 train_evaluations.txt")
print("------------------------------------------")