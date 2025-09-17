# ...existing code...
import json
import os
from typing import Any
import numpy as np
import gymnasium as gym

class TrajectoryRecorder(gym.Wrapper):
    """
    Wrapper that records environment trajectories to a JSONL file.
    """

    def __init__(self, env: gym.Env, log_path: str):
        super().__init__(env)
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.current_obs = None
        self.trajectory = []
        self.ep_id = -1
        self.t = 0

    def reset(self, **kwargs):
        if self.trajectory:
            self._write_episode()
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        self.trajectory = []
        self.t = 0
        self.ep_id += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        record = {
            "episode_id": int(self.ep_id),
            "t": int(self.t),
            "obs": self._to_jsonable(self.current_obs),
            "action": self._to_jsonable(action),
            "reward": float(reward),
            "next_obs": self._to_jsonable(obs),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        self.trajectory.append(record)
        self.current_obs = obs
        self.t += 1

        if terminated or truncated:
            self._write_episode()

        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self._write_episode()
        finally:
            return self.env.close()

    def _write_episode(self):
        if not self.trajectory:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            # ensure_ascii=False 更友好地保留非 ascii 字符
            json.dump(self.trajectory, f, ensure_ascii=False)
            f.write("\n")
        self.trajectory = []

    @staticmethod
    def _to_jsonable(x: Any) -> Any:
        """Convert numpy scalars/arrays and common types to JSON-serializable python types."""
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [TrajectoryRecorder._to_jsonable(v) for v in x]
        # numpy scalar e.g. numpy.int64, numpy.float64
        if isinstance(x, np.generic):
            return x.item()
        # common Python numeric types are fine
        if isinstance(x, (int, float, str, bool)) or x is None:
            return x
        # try to handle objects with __dict__ or fallback to str
        try:
            return float(x) if hasattr(x, "astype") else str(x)
        except Exception:
            return str(x)
