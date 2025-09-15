import os
import json
import yaml

def load_truncated_trajectories(path, max_steps: int = 50) -> str:
    """Load trajectories from a JSONL file and truncate each to the last
    ``max_steps`` entries.

    Each line in the file represents a list of steps for one episode. The
    truncated lists are serialized back to JSON strings and concatenated with
    newlines.

    Parameters
    ----------
    path: str
        Path to the JSONL file.
    max_steps: int, optional
        Maximum number of steps to keep from the end of each trajectory.

    Returns
    -------
    str
        Newline-separated JSON strings of truncated trajectories.
    """

    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            if len(traj) > max_steps:
                traj = traj[-max_steps:]
            texts.append(json.dumps(traj))
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