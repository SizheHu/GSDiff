import os
import numpy as np

# 初始化一个字典来存储所有的 FID 和 KID 值
metrics_dict = {}

# 遍历当前目录下的所有文件夹
for folder in os.listdir('./T'):
    if os.path.isdir('./T/' + folder) and '-' in folder:
        if len(folder.split('-')) == 2 and folder.split('-')[0].isupper() and folder.split('-')[1].isdigit():
            # 构建 test_metrics.npy 文件的完整路径
            file_path = os.path.join('./T/' + folder, 'test_metrics.npy')
            # 检查文件是否存在
            if os.path.isfile(file_path):
                # 加载 numpy 数组
                data = np.load(file_path, allow_pickle=True)[0]
                # 获取 FID 和 KID 值
                fid_value = float(data[1])
                kid_value = float(data[2])
                # 从文件夹名称中获取编号
                group, number = folder.split('-')
                # 将数据添加到字典中
                if number not in metrics_dict:
                    metrics_dict[number] = {'FID': [], 'KID': []}
                metrics_dict[number]['FID'].append(fid_value)
                metrics_dict[number]['KID'].append(kid_value)

# 输出每个编号的 FID 和 KID 平均值
print(f"{'Number':<10}{'FID Average':<20}{'KID Average':<20}")
for number, values in sorted(metrics_dict.items(), key=lambda x: int(x[0])):
    # print(values['FID'], type(values['FID']))
    avg_fid = np.mean(np.array(values['FID']))
    avg_kid = np.mean(np.array(values['KID']))
    print(f"{number:<10}{avg_fid:<20}{avg_kid:<20}")