# -*- coding: utf-8 -*-

"""
Created on Tue Jun 25 20:26:52 2024

@author: Adam
"""

import os
from openai import OpenAI
import analyzer
import yaml

### 导入 配置文件config.yaml
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
client = OpenAI(api_key=OPENAI_CONFIG['DEEPSEEK_KEY'],
                base_url=OPENAI_CONFIG['DEEPSEEK_BASE_URL'])

prompt_analyzer=analyzer.analyzer_all
user_analyzer=analyzer.user

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

#print(response['choices'][0]['text'])
analyzer_output=completion.choices[0].message.content
#print(analyzer_output)

with open(os.path.join(analyzer.data_path,"analyzer_output.txt"),'w') as write_analyzer:
    print(analyzer_output,file=write_analyzer)