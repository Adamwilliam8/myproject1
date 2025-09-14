import os
import yaml
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

# 读取配置文件
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

class Model:
    def __init__(self, env, model_class, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        self.env = env
        self.model = model_class(OPENAI_CONFIG["MODEL_POLICY"], env, **model_kwargs)

    def train(self, total_timesteps, eval_env=None, eval_freq=10000,
              n_eval_episodes=5, reward_threshold=None, save_path="./models/"):
        callbacks = []

        # 计算到下一个 cycle 还需要训练多少步（如果已对齐，可以改为训练一个完整 cycle）
        cycle = total_timesteps
        remainder = self.model.num_timesteps % cycle
        train_steps = cycle - remainder if remainder != 0 else cycle

        # 定期在测试环境里评估 agent 表现
        if eval_env is not None:
            # 当 agent 平均奖励超过某个阈值时，提前停止训练
            if reward_threshold is not None:
                stop_callback = StopTrainingOnRewardThreshold(
                    reward_threshold=reward_threshold, verbose=1
                )
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=os.path.join(save_path, 'best_Eval', ""),
                    log_path=os.path.join(save_path, 'best_Eval', ""),
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    render=False,
                    callback_on_new_best=stop_callback,
                )
                callbacks.append(eval_callback)
            else:
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=os.path.join(save_path, 'best_Eval', ""),
                    log_path=os.path.join(save_path, 'best_Eval', ""),
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    render=False,
                )
                callbacks.append(eval_callback)

        # 定期保存模型
        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=os.path.join(save_path, 'checkpoints', ""),
            name_prefix='rl_model',
        )
        callbacks.append(checkpoint_callback)

        callback = CallbackList(callbacks) if callbacks else None

        print(f"这次要训练{train_steps}步")
        self.model.learn(
            total_timesteps=train_steps,
            callback=callback,
            reset_num_timesteps=False,
        )

    def save(self, path):
        self.model.save(path)

    def load(self, path, tensorboard_log_path=None):
        load_func = getattr(sb3, OPENAI_CONFIG["MODEL_NAME"]).load
        if tensorboard_log_path is not None:
            self.model = load_func(
                path, env=self.env, tensorboard_log=tensorboard_log_path
            )
        else:
            self.model = load_func(path, env=self.env)
        return self  # 关键：返回 self，避免外部拿到 None
    
    def predict(self, observation, deterministic=True):
        return self.model.predict(observation, deterministic=deterministic)
    
    @property
    def num_timesteps(self):
        return self.model.num_timesteps