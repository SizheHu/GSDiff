import os

import shutil

import random


# 1. 创建新路径
new_path = "./rplang-v3-withsemantics"
if not os.path.exists(new_path):
    os.makedirs(new_path)

# 2. 创建txt文件
with open(os.path.join(new_path, "info.txt"), "w") as f:
    f.write("v3与v2的内容相同，但是进行了重新划分")
f.close()

# 3. 在新路径下创建train, val, test文件夹
train_path = os.path.join(new_path, "train")
val_path = os.path.join(new_path, "val")
test_path = os.path.join(new_path, "test")
for path in [train_path, val_path, test_path]:
    if not os.path.exists(path):
        os.makedirs(path)


# 4. 复制文件到新路径
old_train_path = "./rplang-v2-withsemantics/train"
old_test_path = "./rplang-v2-withsemantics/test"

all_files = []
for file in os.listdir(old_train_path):
    shutil.copy(os.path.join(old_train_path, file), new_path)
    all_files.append(file)

for file in os.listdir(old_test_path):
    shutil.copy(os.path.join(old_test_path, file), new_path)
    all_files.append(file)

# 5. 随机划分文件到train, val, test文件夹
random.shuffle(all_files)
train_files = all_files[:65763]
val_files = all_files[65763:68763]
test_files = all_files[68763:]

for file in train_files:
    shutil.move(os.path.join(new_path, file), train_path)

for file in val_files:
    shutil.move(os.path.join(new_path, file), val_path)

for file in test_files:
    shutil.move(os.path.join(new_path, file), test_path)