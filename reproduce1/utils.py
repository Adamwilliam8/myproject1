import os
import json
import yaml
import shutil

def _compact_trajectories(jsonl_text: str, max_steps=50):
    """
    输入: load_truncated_trajectories 返回的多行 JSON 字符串（每行是一个 episode 的 JSON 列表）
    输出: JSON 字符串，包含每个 episode 的最后 max_steps 步的简化字段
    简化字段：episode_id, t, action, reward, ego_speed, ego_lane
    假设 obs 是 list，ego 在 obs[0]，ego_speed 在 obs[0][1]，ego_lane 在 obs[0][2]
    """
    if not jsonl_text:
        return "[]"
    out_eps = []
    for line in jsonl_text.splitlines():
        try:
            ep = json.loads(line)
        except Exception:
            continue
        compact = []
        for step in ep[-max_steps:]:
            obs = step.get("obs")
            ego_speed = None
            ego_lane = None
            try:
                if isinstance(obs, list) and len(obs) > 0 and isinstance(obs[0], (list, tuple)):
                    ego = obs[0]
                    ego_speed = float(ego[1]) if len(ego) > 1 else None
                    ego_lane = float(ego[2]) if len(ego) > 2 else None
            except Exception:
                ego_speed = None
                ego_lane = None
            compact.append({
                "episode_id": int(step.get("episode_id", -1)),
                "t": int(step.get("t", -1)),
                "action": step.get("action"),
                "reward": float(step.get("reward", 0.0)),
                "ego_speed": ego_speed,
                "ego_lane": ego_lane
            })
        out_eps.append(compact)
    return json.dumps(out_eps, ensure_ascii=False)

def load_truncated_trajectories(path, max_steps: int = 50, max_episodes: int = 3) -> str:
    """Load trajectories from a JSONL file, truncate each to the last `max_steps` entries,
    and only keep the last `max_episodes` episodes to limit total size.
    Returns newline-separated JSON strings of truncated trajectories.
    """
    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        return ""

    # 只保留最后 max_episodes 条 episode（若文件很大，这样能显著减小输出）
    selected = lines[-max_episodes:]
    texts = []
    for line in selected:
        try:
            traj = json.loads(line)
        except Exception:
            # 若行不是合法 json，跳过或以原文保留（这里选择跳过）
            continue
        if len(traj) > max_steps:
            traj = traj[-max_steps:]
        texts.append(json.dumps(traj, ensure_ascii=False))
    return "\n".join(texts)


def find_newest_model_dir(models_path):
    ## 找到最新的模型文件夹
    dir_content = os.listdir(models_path)
    version=[]
    for name in dir_content:
        parts=name.split("_")
        if len(parts)>1 and parts[1].isdigit():
            version.append(int(parts[1]))
    data_path = os.path.join(models_path, 'DQN_'+str(max(version)), '')
    return data_path

def copy_directory(src, dst):
    """复制目录内容，如果源不存在则创建空目标目录。"""
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        if not os.path.exists(dst):
            os.makedirs(dst, exist_ok=True)


def copy_file(src, dst):
    """复制文件并在需要时创建父目录。"""
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)