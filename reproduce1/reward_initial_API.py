# -*- coding: utf-8 -*-
"""
# @Time    : 2024/6/29 16:20
# @Author  : Adam
"""

from openai import OpenAI
import reward_initial
import yaml
import json

### 导入 配置文件config.yaml
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
client = OpenAI(api_key=OPENAI_CONFIG['DEEPSEEK_KEY'],
                base_url=OPENAI_CONFIG['DEEPSEEK_BASE_URL'])

prompt_reward_initial=reward_initial.reward_initial_prompt
user_reward_initial=reward_initial.user_reward_initial
messages=[
    {"role": "system", "content": prompt_reward_initial},
    {"role": "user", "content": user_reward_initial}
  ]


# 继续让模型生成下一步内容
completion = client.chat.completions.create(
  model=OPENAI_CONFIG['DEEPSEEK_MODEL'],
  messages=messages,
  max_tokens = OPENAI_CONFIG['DEEPSEEK_MAX_TOKENS'],
  temperature = OPENAI_CONFIG['DEEPSEEK_TEMPERATURE'],
  stream=False
)

print(completion)
if hasattr(completion, "choices") and completion.choices:
    print("返回内容：", completion.choices[0].message.content)
else:
    print("API 没有返回有效内容")
    
#print(response['choices'][0]['text'])
reward_initial_output=completion.choices[0].message.content
#print(reward_initial_output)

with open('reward_initial_output.txt','w') as write_reward_initial:
    #print('# -*- coding: utf-8 -*-')
    print(reward_initial_output,file=write_reward_initial)
    
file = open('reward_initial_output.txt','r')
reward_function_lines = file.readlines()
reward_function = ''
for item in reward_function_lines:
    if "```python" in item or "```" in item:
        pass
    else:
        reward_function = reward_function + item
file.close()

with open('reward_function.py','w') as write_reward:
    #print('# -*- coding: utf-8 -*-')
    print(reward_function,file=write_reward)
    
with open('reward_function_initial.py','w') as write_reward:
    #print('# -*- coding: utf-8 -*-')
    print(reward_function,file=write_reward)