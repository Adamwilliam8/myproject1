from model import Model
import highway_env
import gymnasium as gym
import stable_baselines3 as sb3
import numpy as np
import os
from visualization import Visualization
import yaml
from trajectory_logger import TrajectoryRecorder
from utils import find_newest_model_dir

# 读取配置文件
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
# 传递给 正在用的 的模型名称
model_name = os.environ.get('REWARD_MODEL_NAME', 'unknown')

## 加载环境和模型
env =  gym.make(
    OPENAI_CONFIG["ENV_ID"],
    max_episode_steps=OPENAI_CONFIG["MAX_EPISODE_STEPS"]
)
from reward_function import _reward
env._reward = _reward.__get__(env, type(env))  # 绑定为实例方法
env = TrajectoryRecorder(env, "test_trajectories.jsonl")
obs, info = env.reset()

# 把新写的奖励函数 绑定到当前的env实例上，覆盖掉原来类里的_reward方法
from reward_function import _reward
env._reward = _reward.__get__(env, type(env))  # 绑定为实例方法

tensorboard_log_path = OPENAI_CONFIG["tensorboard_log_path"]
model = getattr(sb3, OPENAI_CONFIG["MODEL_NAME"]).load(os.path.join(tensorboard_log_path, "final_model.zip"),env=env)


data_path = find_newest_model_dir(OPENAI_CONFIG["tensorboard_log_path"])

Visualization = Visualization(
    path=data_path, 
    dpi=96
)

### 创造/覆盖文件 里面写入 测试的奖励函数(之后给ai分析用)
with open('test_evaluations.txt', 'w') as f:
    f.write(f"测试评估结果（模型: {model_name}):\n\n")

### 测试 强化学习模型
step_count = 0
max_steps = 1000
episode_count = 0

total_rewards = []
episode_rewards = []
cumulative_rewards = []

cumulative_reward = 0
while step_count < max_steps:
    done = truncated = False
    obs, info = env.reset()
    episode_count += 1
    episode_reward = 0
    
    while not (done or truncated) and step_count < max_steps:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # 记录关键数据
        if step_count % 50 == 0:  # 每50步记录一次
            total_rewards.append(reward)
            
            with open('test_evaluations.txt', 'a') as f:
                f.write(f"步数{step_count}: 总奖励={reward:.3f}, 速度={env.unwrapped.vehicle.speed:.1f}, "
                        f"车道={env.unwrapped.vehicle.lane_index[2]}, 碰撞={env.unwrapped.vehicle.crashed}\n")
        
        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)

        episode_reward += reward
        step_count += 1
    
    with open('test_evaluations.txt', 'a') as f:
        f.write(f"Episode {episode_count}结束: 奖励={episode_reward:.3f}, "
                f"{'碰撞' if done and info.get('crashed',False) else '训练时间到' if truncated else '成功了' if done else '进行中'}\n")
        f.write("---------------------------------------------------\n")
        f.write(f"done={done},truncated={truncated},info={info}\n")
        f.write("---------------------------------------------------\n")
    episode_rewards.append(episode_reward)

# 分析奖励组件
mean_reward = np.mean(total_rewards)
with open('test_evaluations.txt', 'a') as f:
    f.write(f"\n=== 奖励组件分析 ===\n")
    f.write(f"总奖励 - 平均: {mean_reward:.3f}, 标准差: {np.std(total_rewards):.3f}")

with open('reward_model_scores.txt', 'a') as score_file:
    score_file.write(f"{model_name},{mean_reward:.3f}\n")

print("转换完成，内容已写入 test_evaluations.txt")


Visualization.save_data_and_plot(data=total_rewards,  # 记录步数(能被50整除)的单个reward
                                 filename='every_50_steps_reward', 
                                 xlabel='every 50 steps', 
                                 ylabel='every 50 steps reward')
Visualization.save_data_and_plot(data=episode_rewards,  # 记录每个episode的总reward
                                 filename='every_episode_reward', 
                                 xlabel='every episode', 
                                 ylabel='every episode reward')
Visualization.save_data_and_plot(data=cumulative_rewards,  # 记录每步累加reward
                                 filename='cumulative_reward', 
                                 xlabel='every step', 
                                 ylabel='cumulative reward for agent')

env.close()