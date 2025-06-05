import sys
sys.path.append('/home/user00/HSZ/gsdiff-main')
sys.path.append('/home/user00/HSZ/gsdiff-main/datasets')
sys.path.append('/home/user00/HSZ/gsdiff-main/gsdiff')
sys.path.append('/home/user00/HSZ/gsdiff-main/scripts/metrics')


import math
import torch
import shutil
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from itertools import cycle
from datasets.rplang_edge_semantics_simplified import RPlanGEdgeSemanSimplified
from gsdiff.utils import *
import torch.nn.functional as F
from scripts.metrics.fid import fid
from scripts.metrics.kid import kid

import os


def deep_compare(a, b):
    '''判断两个字典数据是否完全相同'''
    # 如果类型不同，直接返回 False
    if type(a) != type(b):
        return False

    # 如果是字典，比较它们的 key 与对应的值
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(deep_compare(a[key], b[key]) for key in a)

    # 如果是列表或元组，则逐项比较
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_compare(item1, item2) for item1, item2 in zip(a, b))

    # 如果是 numpy 数组，则使用 np.array_equal 比较
    elif isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    # 其他类型直接比较（例如 int, float, str 等）
    else:
        return a == b

def check_subdir_file_counts(base_dir):
    subdirs = ['val']
    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        file_list = [f for f in os.listdir(full_path)
                     if os.path.isfile(os.path.join(full_path, f))]
        print('文件数' + str(len(file_list)))


if not os.path.exists('rplandata/Data/rplang-v3-bubble-diagram'):
    os.mkdir('./rplandata/Data/rplang-v3-bubble-diagram')
os.mkdir('./rplandata/Data/rplang-v3-bubble-diagram/val')


bubblecategory = {}
bubblecategory[0] = 0
bubblecategory[1] = 0
bubblecategory[2] = 0
bubblecategory[3] = 0
bubblecategory[4] = 0
bubblecategory[5] = 0
edgecategory = {}
edgecategory[0] = 0
edgecategory[1] = 0
bubblenumber = {}
val_files = os.listdir('rplandata/Data/rplang-v3-withsemantics/val')
for val_file in tqdm(val_files):
    val_graph = np.load('rplandata/Data/rplang-v3-withsemantics/val/' + val_file, allow_pickle=True).item()

    '''file_id
        corners
        adjacency_matrix
        adjacency_list
        corners_np
        adjacency_matrix_np
        adjacency_list_np
        corner_list_np_normalized
        corner_list_np_normalized_padding
        padding_mask
        global_matrix_np_padding
        adjacency_matrix_np_padding
        edge_coords
        edges
        semantics
        corner_list_np_normalized_padding_withsemantics
        '''
    '''coords_withsemantics, (53, 16)'''
    corners_withsemantics = val_graph['corner_list_np_normalized_padding_withsemantics']
    # 初始化一个n*9的新数组(53, 9)
    corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
    # 复制第0、1列
    corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
    # 计算新的第2列
    corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
    # 计算新的第3列
    corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
    # 计算新的第4列
    corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
    # 复制第4、5、11、15列
    corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4]
    corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5]
    corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]
    corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15]

    '''attn 1 matrix, (53, 53)'''
    global_attn_matrix = val_graph['global_matrix_np_padding'].astype(bool)
    '''corners padding mask, (53, 1)'''
    corners_padding_mask = val_graph['padding_mask']

    '''edges, (2809, 1)'''
    edges = val_graph['edges']
    corners_withsemantics_0_val = corners_withsemantics_simplified[None, :, :]
    global_attn_matrix_val = global_attn_matrix[None, :, :]
    corners_padding_mask_val = corners_padding_mask[None, :, :]
    edges_val = edges[None, :, :]
    corners_withsemantics_0_val = corners_withsemantics_0_val.clip(-1, 1)
    corners_0_val = (corners_withsemantics_0_val[0, :, :2] * 128 + 128).astype(int)
    semantics_0_val = corners_withsemantics_0_val[0, :, 2:].astype(int)
    global_attn_matrix_val = global_attn_matrix_val
    corners_padding_mask_val = corners_padding_mask_val
    edges_val = edges_val
    corners_0_val_depadded = corners_0_val[corners_padding_mask_val.squeeze() == 1][None, :, :]  # (n, 2)
    semantics_0_val_depadded = semantics_0_val[corners_padding_mask_val.squeeze() == 1][None, :, :]  # (n, 7)
    edges_val_depadded = edges_val[global_attn_matrix_val.reshape(1, -1, 1)][None, :, None]
    edges_val_depadded = np.concatenate((1 - edges_val_depadded, edges_val_depadded), axis=2)

    ''' get planar cycles'''
    # 形状为 (1, n, 14) 的 ndarray，包含 0 和 1;找到每个子数组中 1 所在的索引,用 99999 替换值为 0 的原始元素
    semantics_gt_i_transform_val = semantics_0_val_depadded
    semantics_gt_i_transform_indices_val = np.indices(semantics_gt_i_transform_val.shape)[-1]
    semantics_gt_i_transform_val = np.where(semantics_gt_i_transform_val == 1,
                                              semantics_gt_i_transform_indices_val, 99999)

    gt_i_points_val = [tuple(corner_with_seman_val) for corner_with_seman_val in
                         np.concatenate((corners_0_val_depadded, semantics_gt_i_transform_val), axis=-1).tolist()[
                             0]]
    # print(output_points)
    gt_i_edges_val = edges_to_coordinates(
        np.triu(edges_val_depadded[0, :, 1].reshape(len(gt_i_points_val), len(gt_i_points_val))).reshape(-1),
        gt_i_points_val)

    # print(gt_i_points_val)
    # print(gt_i_edges_val)

    '''我们当初做实验的时候，气泡图gt的语义的提取具有随机性，但是根据RPLAN数据集的条款，我们没有权利以任何方式公开RPLAN数据集的内容。
    我们能提供的只有这个提取气泡图数据的脚本。
    所以你们用相同的脚本提取的气泡图gt的语义与我们自己做实验的时候会有部分不同，但是因为数据集规模较大，最终的性能指标在统计意义上应该不会有太大差异。
    你们也可以使用get_cycle_basis_and_semantic_3_semansimplified来提取房间语义并自己训练拓扑图相关的模型，get_cycle_basis_and_semantic_3_semansimplified这个方法不是随机的，相比论文中的指标可能会有提升。'''
    d_rev_val, simple_cycles_val, simple_cycles_semantics_val = get_cycle_basis_and_semantic_2_semansimplified(
        gt_i_points_val,
        gt_i_edges_val)
    simple_cycles_val_ = []
    for sc in simple_cycles_val:
        sc_val = [(t[0], t[1]) for t in sc]
        simple_cycles_val_.append(sc_val)
        # print(sc_val)
    # for scs in simple_cycles_semantics_val:
    #     print(scs)
    polygons = simple_cycles_val_
    edges = [[(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))][:-1] for polygon in polygons]
    # print(edges)


    def get_adjacency_matrix(polygons):
        n = len(polygons)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if any(set(edge) in [set(edge_j) for edge_j in polygons[j]] for edge in polygons[i]):
                    matrix[i][j] = 1
                    matrix[j][i] = 1
        return matrix


    adjacency_matrix = get_adjacency_matrix(edges)
    edgecategory[1] += np.sum(np.triu(np.array(adjacency_matrix)))
    edgecategory[0] += ((len(adjacency_matrix) * (len(adjacency_matrix) + 1)) / 2) - np.sum(np.triu(np.array(adjacency_matrix)))


    # 使用Shoelace公式来计算面积，然后使用重心的公式来计算凹多边形的重心。在考虑邻接性并绘制时，我们可以使用OpenCV图形库。以下是相关的Python代码：
    # 计算多边形的重心
    def get_polygon_centroid(polygon):
        area = 0
        x = 0
        y = 0
        for i in range(-1, len(polygon) - 1):
            step = (polygon[i][0] * polygon[i + 1][1]) - (polygon[i + 1][0] * polygon[i][1])
            area += step
            x += (polygon[i][0] + polygon[i + 1][0]) * step
            y += (polygon[i][1] + polygon[i + 1][1]) * step
        area /= 2
        x /= (6 * area)
        y /= (6 * area)
        return (int(x), int(y))




    # 计算每个多边形的重心
    centroids = [get_polygon_centroid(polygon[:-1]) for polygon in polygons]


    # 保存气泡图的房间多边形（首尾相同）、重心、类型、邻接矩阵。
    bbdiagram = {}
    bbdiagram['file_id'] = val_graph['file_id']
    bbdiagram['polygons'] = simple_cycles_val
    bbdiagram['centroids'] = centroids
    bbdiagram['semantics'] = simple_cycles_semantics_val
    
    for s in simple_cycles_semantics_val:
        bubblecategory[s] += 1

    bbdiagram['adjacency_matrix'] = adjacency_matrix
    bbdiagram['corner_number'] = len(val_graph['corners'])
    
    
    np.save(os.path.join('rplandata/Data/rplang-v3-bubble-diagram/val', f"{val_graph['file_id']}.npy"), bbdiagram)

