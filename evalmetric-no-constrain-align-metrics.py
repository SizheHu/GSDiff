import os
import numpy as np
import torch
from tqdm import *

batch_size = 1
device = 'cpu'

# 初始化一个字典来存储所有的 FID 和 KID 值以及对齐损失
metrics_dict = {}

# 遍历当前目录下的所有文件夹
for folder in os.listdir('./T'):
    if os.path.isdir('./T/' + folder) and '-' in folder:
        if len(folder.split('-')) == 2 and folder.split('-')[0].isupper() and folder.split('-')[1].isdigit():
            # 该实验编号该组的3000样本的平均对齐损失
            avg_align_loss = 0
            # 遍历vr4stat_i.npy文件
            for i in tqdm(range(3000)):
                file_name = f"vr4stat_{i}.npy"
                file_path = os.path.join('./T/' + folder, file_name)

                # 检查文件是否存在
                if os.path.isfile(file_path):
                    # 加载 numpy 数组
                    data = np.load(file_path, allow_pickle=True).item()
                    # 提取'output_points_test'键对应的值并转为(1, 53, 2)（后面有的值是99999）
                    output_points_test = torch.tensor(np.array([list(tupl)[:2] for tupl in data['output_points_test']])[None, :, :])

                    pred_xstart_cs = torch.zeros((1, 53, 10))
                    pred_xstart_cs[:, 0:output_points_test.shape[1], 0:2] = output_points_test
                    pred_xstart_cs[:, output_points_test.shape[1]:, 9:10] = 1

                    # 根据最后一列是否为0来筛选出对应的行
                    mask = pred_xstart_cs[:, :, -1] == 0

                    # 仅选择最后一列为0的行的前两列（x和y坐标）
                    x_coords = pred_xstart_cs[:, :, 0] * mask
                    y_coords = pred_xstart_cs[:, :, 1] * mask

                    # 设置mask中的False值为无穷大，以便在计算距离时排除这些值
                    inf_mask = torch.where(mask, 0, float('inf'))
                    x_coords += inf_mask
                    y_coords += inf_mask

                    # 计算 x 坐标的 L1 距离方阵
                    x_coords_uns = x_coords.unsqueeze(2)
                    distance_matrix_x = torch.abs(x_coords_uns - x_coords_uns.transpose(1, 2))

                    # 计算 y 坐标的 L1 距离方阵
                    y_coords_uns = y_coords.unsqueeze(2)
                    distance_matrix_y = torch.abs(y_coords_uns - y_coords_uns.transpose(1, 2))

                    # 设置无穷大和 NaN 值为99999
                    distance_matrix_x[torch.isinf(distance_matrix_x) | torch.isnan(distance_matrix_x)] = 99999
                    distance_matrix_y[torch.isinf(distance_matrix_y) | torch.isnan(distance_matrix_y)] = 99999
                    # 对角线使用掩码填充99999
                    distance_matrix_x[
                        (torch.eye(53).unsqueeze(0) == 1).to(device).expand(distance_matrix_x.size(0), 53, 53)] = 99999
                    distance_matrix_y[
                        (torch.eye(53).unsqueeze(0) == 1).to(device).expand(distance_matrix_y.size(0), 53, 53)] = 99999
                    # print(distance_matrix_x)
                    # print(distance_matrix_y)
                    # 计算每个节点 i 的所有 x, y 轴向距离的最小值
                    min_values_x, _ = torch.min(distance_matrix_x, dim=2)
                    min_values_y, _ = torch.min(distance_matrix_y, dim=2)
                    # print(min_values_x.shape) # (bs, 53)
                    # print(min_values_y.shape) # (bs, 53)
                    # LACE原文：求同一个节点i对其他节点（53）的所有方向（2）中距离（106）最小的
                    min_values = torch.stack((min_values_x, min_values_y), dim=2)
                    min_values, _ = torch.min(min_values, dim=2)
                    # print(min_values.shape) # (bs, 53) 包含99999,那代表节点为padding节点
                    # 对于每个样本，将非99999的元素套上g=-2log(1-(0.5-eps)x)

                    # 应用函数 -2log(1-0.5x) 到非99999的元素
                    # 注意：确保输入到log的值是正数，即1-0.5x > 0
                    # 99999这里会直接变成0，没有影响
                    # 再乘以时间权重
                    masked_tensor = torch.where(min_values != 99999,
                                                min_values,
                                                torch.tensor(0.0, dtype=min_values.dtype, device=device)).sum(1)
                    # print(masked_tensor.shape)
                    # print(masked_tensor)

                    # 对结果求和
                    avg_align_loss += masked_tensor.sum()

            avg_align_loss /= 3000

            # 从文件夹名称中获取编号
            group, number = folder.split('-')
            # 将数据添加到字典中
            if number not in metrics_dict:
                metrics_dict[number] = {'Alg': []}
            metrics_dict[number]['Alg'].append(avg_align_loss)
# 输出每个编号平均值
print(f"{'Number':<10}{'Alg':<20}")
for number, values in sorted(metrics_dict.items(), key=lambda x: int(x[0])):
    avg_alg = np.mean(np.array(values['Alg']))
    print(f"{number:<10}{avg_alg:<20}")





