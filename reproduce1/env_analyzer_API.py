# -*- coding: utf-8 -*-
"""
# @Time    : 2024/9/17 16:20
# @Author  : Adam
"""
from openai import OpenAI
import env_analyzer
import yaml
import json
import os

### 导入 配置文件config.yaml
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
client = OpenAI(api_key=OPENAI_CONFIG['DEEPSEEK_KEY'],
                base_url=OPENAI_CONFIG['DEEPSEEK_BASE_URL'])

prompt_analyzer=env_analyzer.env_analyzer
user_analyzer=env_analyzer.user_env_analyzer

completion = client.chat.completions.create(
  model=OPENAI_CONFIG['DEEPSEEK_MODEL'],
  messages=[
    {"role": "system", "content": prompt_analyzer},
    {"role": "user", "content": user_analyzer}
  ],
  max_tokens = OPENAI_CONFIG['DEEPSEEK_MAX_TOKENS'],
  temperature = OPENAI_CONFIG['DEEPSEEK_TEMPERATURE'],
  stream=False
)

print(f"模型输入消耗的tokens数:{completion.usage.prompt_tokens}")
print(f"模型输出消耗的tokens数:{completion.usage.completion_tokens}")

env_analyzer_output=completion.choices[0].message.content

with open("env_analyzer_output.txt",'w') as write_analyzer:
    print(env_analyzer_output,file=write_analyzer)
