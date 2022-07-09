"""
This is a script for collecting all manualled labels in on pitch in label_making file.
"""
import re; import os; import os.path as osp;
data = []
pattern = r'speed_(.*)_(.*)_(.*).txt'
for file in os.listdir():
    if file.startswith('speed') and file.endswith('txt'):
        ret = re.search(pattern, file)
        frame = ret.groups()[2]
        with open(file, 'r') as f:
            line = f.readline().strip()
            line += f' 1 {frame}\n'
            data.append(line)
            print(line)
with open('fake.txt', 'w') as f:
    f.writelines(data)