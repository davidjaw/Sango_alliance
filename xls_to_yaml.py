import yaml
import pandas as pd

df = pd.read_excel('4808.xls', header=0)

idx = 0
data = []
for index, row in df.iterrows():
    obj = {
        'unicode': row[2],
        'index': row[0] - 1,
        'content': row[3]
    }
    data.append(obj)
    idx += 1

for i, letter in enumerate(range(ord('A'), ord('Z') + 1)):
    data.append({
        'index': idx,
        'unicode': hex(letter),
        'content': chr(letter)
    })
    idx += 1

for i, letter in enumerate(range(ord('a'), ord('z') + 1)):
    data.append({
        'index': idx,
        'unicode': hex(letter),
        'content': chr(letter)
    })
    idx += 1

with open('data.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data, f, allow_unicode=True)
