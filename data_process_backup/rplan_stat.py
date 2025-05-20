import numpy as np
import os
from tqdm import *

# 定义主目录
base_dir = './rplang-v3-bubble-diagram'

# 定义子目录列表
sub_dirs = ['train', 'val', 'test']

# 创建一个字典来保存数据
dataroom = {0: 0, 1: 0, 2: 0, 3:0, 4:0, 5:0}
dataroomnum = {4:0, 5:0, 6:0, 7:0, 8:0}
datacornernum = {}
dataedgenum = {}

# 遍历每个子目录
for sub_dir in sub_dirs:
    # 获取子目录的完整路径
    dir_path = os.path.join(base_dir, sub_dir)

    # 确保目录存在
    if not os.path.isdir(dir_path):
        print(f"Directory does not exist: {dir_path}")
        continue

    # 遍历目录中的所有文件
    for file in tqdm(os.listdir(dir_path)):
        # 检查文件扩展名是否为.npy
        if file.endswith('.npy'):
            # 读取.npy文件
            file_path = os.path.join(dir_path, file)
            array = np.load(file_path, allow_pickle=True).item()


            semantics = array['semantics']
            for s in semantics:
                dataroom[s] += 1
            dataroomnum[len(semantics)] += 1
            if array['corner_number'] not in datacornernum.keys():
                datacornernum[array['corner_number']] = 1
            else:
                datacornernum[array['corner_number']] += 1
            array2 = np.load(file_path.replace('bubble-diagram', 'withsemantics'), allow_pickle=True).item()
            edgenumber = np.triu(array2['edges'].reshape(53, 53)).sum().item()
            if edgenumber not in dataedgenum.keys():
                dataedgenum[edgenumber] = 1
            else:
                dataedgenum[edgenumber] += 1
print(dataroom)
print(dataroomnum)
print(datacornernum)
print(dataedgenum)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Set global font size
plt.rcParams.update({'font.size': 17})  # You can change '14' to your desired size


# 房间类型的映射
room_type_mapping = {0: 'living room', 1: 'bedroom', 2: 'storage', 3: 'kitchen', 4: 'bathroom', 5: 'balcony'}

# 更新并排序dataroom字典的键
dataroom = {room_type_mapping[k]: v for k, v in sorted(dataroom.items())}

# 对其他字典按键进行排序
dataroomnum = dict(sorted(dataroomnum.items()))
datacornernum = dict(sorted(datacornernum.items()))
dataedgenum = dict(sorted(dataedgenum.items()))

# 定义一个函数来绘制柱状图，并设置刻度标签的旋转和间隔
def plot_bar_chart(data, title, x_label, y_label, rotation=45, width=10, height=6, color='skyblue', tick_spacing=2):
    # 创建图表
    plt.figure(figsize=(width, height))
    plt.bar(list(data.keys()), list(data.values()), color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 设置x轴的刻度标签和间隔
    ax = plt.gca()
    ax.set_xticks(list(data.keys()))
    ax.set_xticklabels(list(data.keys()), rotation=rotation)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    plt.tight_layout()  # 调整布局以适应标签
    plt.show()

# 使用不同的颜色和色调来绘制图表
plot_bar_chart(dataroom, 'Number of Different Room Types', 'Room Type', 'Count', rotation=0, color='lightblue', tick_spacing=1)
plot_bar_chart(dataroomnum, 'Number of Rooms', 'Number of Rooms', 'Count', rotation=0, color='cyan', tick_spacing=1)
plot_bar_chart(datacornernum, 'Number of Corners', 'Number of Corners', 'Count', rotation=0, color='lightgreen', tick_spacing=2)
plot_bar_chart(dataedgenum, 'Number of Edges', 'Number of Edges', 'Count', rotation=0, color='lightcoral', tick_spacing=2)
