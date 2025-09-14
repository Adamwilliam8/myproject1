# -*- coding: utf-8 -*-
"""
# @Time    : 2024/6/29 16:20
# @Author  : Adam
"""
import highway_env
from openai import OpenAI
import reward_initial
import yaml
import json
import os

# 导入 配置文件config.yaml
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
# 支持的模型配置
MODEL_CONFIGS = {
    'deepseek': {
        'key': OPENAI_CONFIG['DEEPSEEK_KEY'],
        'base_url': OPENAI_CONFIG['DEEPSEEK_BASE_URL'],
        'model': OPENAI_CONFIG['DEEPSEEK_MODEL'],
        'max_tokens': OPENAI_CONFIG['DEEPSEEK_MAX_TOKENS'],
        'temperature': OPENAI_CONFIG['DEEPSEEK_TEMPERATURE'],
    },
    'qwen': {
        'key': OPENAI_CONFIG['QWEN_KEY'],
        'base_url': OPENAI_CONFIG['QWEN_BASE_URL'],
        'model': OPENAI_CONFIG['QWEN_MODEL'],
        'max_tokens': OPENAI_CONFIG['QWEN_MAX_TOKENS'],
        'temperature': OPENAI_CONFIG.get('QWEN_TEMPERATURE', 0.7),
    },
}

"""调用指定模型并返回 对应的奖励函数代码"""
def call_llm(model_name: str, config: dict, messages):
    client = OpenAI(api_key=config['key'], base_url=config['base_url'])
    completion = client.chat.completions.create(
        model=config['model'],
        messages=messages,
        max_tokens=config['max_tokens'],
        temperature=config.get('temperature', 0.3),
        stream=False,
    )

    if hasattr(completion, "choices") and completion.choices:
        reward_initial_output = completion.choices[0].message.content
    else:
        print(f"{model_name} API 没有返回有效内容")
        return

    output_dir = os.path.join('rewards', model_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'reward_initial_output.txt'), 'w') as f:
        print(reward_initial_output, file=f)

    with open(os.path.join(output_dir, 'reward_initial_output.txt'), 'r') as file:
        reward_function_lines = file.readlines()

    reward_function = ''
    for item in reward_function_lines:
        if "```python" in item or "```" in item:
            continue
        reward_function += item

    with open(os.path.join(output_dir, 'reward_function.py'), 'w') as write_reward:
        print(reward_function, file=write_reward)


prompt_reward_initial = reward_initial.reward_initial_prompt
user_reward_initial = reward_initial.user_reward_initial
messages = [
    {"role": "system", "content": prompt_reward_initial},
    {"role": "user", "content": user_reward_initial},
]

for name, cfg in MODEL_CONFIGS.items():
    call_llm(name, cfg, messages)


