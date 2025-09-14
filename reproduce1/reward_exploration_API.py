# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:20:26 2024

@author: Adam
"""


import os
from openai import OpenAI
import reward_exploration
import yaml

### 导入 配置文件config.yaml
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
client = OpenAI(api_key=OPENAI_CONFIG['DEEPSEEK_KEY'],
                base_url=OPENAI_CONFIG['DEEPSEEK_BASE_URL'])

prompt_reward_exploration=reward_exploration.reward_exploration_final
user_reward_exploration=reward_exploration.user_reward_exploration

completion = client.chat.completions.create(
  model=OPENAI_CONFIG['DEEPSEEK_MODEL'],
  messages=[
    {"role": "system", "content": prompt_reward_exploration},
    {"role": "user", "content": user_reward_exploration}
  ],
  max_tokens = OPENAI_CONFIG['DEEPSEEK_MAX_TOKENS'],
  temperature = OPENAI_CONFIG['DEEPSEEK_TEMPERATURE'],
  stream=False
)

#print(response['choices'][0]['text'])
reward_exploration_output=completion.choices[0].message.content
#print(reward_modify_output)

with open(reward_exploration.data_path + 'reward_exploration_output.txt','w') as write_reward_exploration:
    #print('# -*- coding: utf-8 -*-')
    print(reward_exploration_output,file=write_reward_exploration)
    
file = open(reward_exploration.data_path + 'reward_exploration_output.txt','r')
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




