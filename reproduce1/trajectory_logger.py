import json
import os
import numpy as np
import gymnasium as gym


class TrajectoryRecorder(gym.Wrapper):
    """Wrapper that records environment trajectories to a JSONL file."""

    def __init__(self, env: gym.Env, log_path: str):
        super().__init__(env)
        self.log_path = log_path
        # ensure directory exists if log_path includes directory
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.current_obs = None
        self.trajectory = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        self.trajectory = []
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # store data in serialisable format
        record = {
            "obs": self._to_list(self.current_obs),
            "action": self._to_list(action),
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
        }
        self.trajectory.append(record)
        self.current_obs = obs

        if done or truncated:
            self._write_episode()
        return obs, reward, done, truncated, info

    def _write_episode(self):
        if not self.trajectory:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            json.dump(self.trajectory, f)
            f.write("\n")
        self.trajectory = []

    @staticmethod
    def _to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return list(x)
        return x