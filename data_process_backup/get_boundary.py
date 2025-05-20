import sys

import numpy as np

sys.path.append('/home/user00/HSZ/house_diffusion-main')
sys.path.append('/home/user00/HSZ/house_diffusion-main/datasets')
sys.path.append('/home/user00/HSZ/house_diffusion-main/house_diffusion')
sys.path.append('/home/user00/HSZ/house_diffusion-main/scripts/metrics')


import math
import torch
import shutil
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from itertools import cycle
from datasets.rplang_edge_semantics_simplified import RPlanGEdgeSemanSimplified
from house_diffusion.heterhouse_55_3 import HeterHouseModel
from house_diffusion.heterhouse_56_2 import EdgeModel
from house_diffusion.utils import *
import torch.nn.functional as F
from scripts.metrics.fid import fid
from scripts.metrics.kid import kid
import copy


diffusion_steps = 1000
lr = 1e-4
weight_decay = 0
total_steps = float("inf")
device = 'cuda:0'
merge_points = True

import os

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

boundary_num = [] # max: 38 (76 dim)

'''前置数据：rplang-v3-withsemantics'''
'''输出数据：rplang-v3-withsemantics-withboundary和rplang-v3-withsemantics-withboundary-v2'''

train_path = '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-withsemantics'

train_files = os.listdir(train_path)
for train_file in tqdm(train_files):
    # print(train_file)
    train_graph = np.load(train_path + '/' + train_file, allow_pickle=True).item()

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
    '''coords_withsemantics, (53, 18)'''
    corners_withsemantics = train_graph['corner_list_np_normalized_padding_withsemantics']
    # 初始化一个n*9的新数组(53, 9)
    corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
    # 复制第0、1列 坐标
    corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
    # 计算新的第2列 标签0 客厅/餐厅/玄关
    corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 3, 12, 15, 16]]).sum(axis=1)
    # 计算新的第3列 标签1 卧室/书房
    corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [4, 5, 6, 9, 10]]).sum(axis=1)
    # 计算新的第4列 标签2 橱柜
    corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
    # 其他
    corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 7] # 标签3 厨房
    corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 8] # 标签4 卫生间
    corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11] # 标签5 阳台
    corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 17] # 标签6 外部

    '''attn 1 matrix, (53, 53)'''
    global_attn_matrix = train_graph['global_matrix_np_padding'].astype(bool)
    '''corners padding mask, (53, 1)'''
    corners_padding_mask = train_graph['padding_mask']

    '''edges, (2809, 1)'''
    edges = train_graph['edges']
    corners_withsemantics_0_train = corners_withsemantics_simplified[None, :, :]
    global_attn_matrix_train = global_attn_matrix[None, :, :]
    corners_padding_mask_train = corners_padding_mask[None, :, :]
    edges_train = edges[None, :, :]
    corners_withsemantics_0_train = corners_withsemantics_0_train.clip(-1, 1)
    corners_0_train = (corners_withsemantics_0_train[0, :, :2] * 128 + 128).astype(int)
    semantics_0_train = corners_withsemantics_0_train[0, :, 2:].astype(int)
    global_attn_matrix_train = global_attn_matrix_train
    corners_padding_mask_train = corners_padding_mask_train
    edges_train = edges_train
    corners_0_train_depadded = corners_0_train[corners_padding_mask_train.squeeze() == 1][None, :, :]  # (n, 2)
    semantics_0_train_depadded = semantics_0_train[corners_padding_mask_train.squeeze() == 1][None, :, :]  # (n, 7)
    edges_train_depadded = edges_train[global_attn_matrix_train.reshape(1, -1, 1)][None, :, None]
    edges_train_depadded = np.concatenate((1 - edges_train_depadded, edges_train_depadded), axis=2)

    ''' get planar cycles'''
    # 形状为 (1, n, 14) 的 ndarray，包含 0 和 1;找到每个子数组中 1 所在的索引,用 99999 替换值为 0 的原始元素
    semantics_gt_i_transform_train = semantics_0_train_depadded
    semantics_gt_i_transform_indices_train = np.indices(semantics_gt_i_transform_train.shape)[-1]
    semantics_gt_i_transform_train = np.where(semantics_gt_i_transform_train == 1,
                                              semantics_gt_i_transform_indices_train, 99999)

    gt_i_points_train = [tuple(corner_with_seman_train) for corner_with_seman_train in
                         np.concatenate((corners_0_train_depadded, semantics_gt_i_transform_train), axis=-1).tolist()[
                             0]]
    # print(output_points)
    gt_i_edges_train = edges_to_coordinates(
        np.triu(edges_train_depadded[0, :, 1].reshape(len(gt_i_points_train), len(gt_i_points_train))).reshape(-1),
        gt_i_points_train)

    # print(gt_i_points_train)
    # print(gt_i_edges_train)

    d_rev_train, simple_cycles_train, simple_cycles_semantics_train = get_cycle_basis_and_semantic_2_semansimplified_4extractingboundary(
        gt_i_points_train,
        gt_i_edges_train)
    simple_cycles_train_ = []
    for sc in simple_cycles_train:
        sc_train = [(t[0], t[1]) for t in sc]
        simple_cycles_train_.append(sc_train)
        # print(sc_train)
    # for scs in simple_cycles_semantics_train:
    #     print(scs)
    polygons = simple_cycles_train_
    # for p in polygons:
    #     print(p)


    # 定义一个函数来计算两个向量之间的角度
    def angle(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.degrees(np.arccos(dot_product / norm_product))


    non_flat_angles = []
    count = 0
    for polygon_i, polygon in enumerate(polygons):
        if simple_cycles_semantics_train[polygon_i] == 6:
            # 多边形的顶点
            polygon.pop(-1)
            # 遍历多边形的每个角
            for i in range(len(polygon)):
                p1, p2, p3 = np.array(polygon[i % len(polygon)]), np.array(polygon[(i + 1) % len(polygon)]), np.array(polygon[(i + 2) % len(polygon)])
                # print(p1, p2, p3)
                v1, v2 = p1 - p2, p3 - p2
                ang = angle(v1, v2)
                # print(ang)
                # 如果角度接近180度（误差在10度之内），打印这个点
                if np.abs(ang - 180) < 10:
                    # print("Flat angle at point:", p2)
                    pass
                else:
                    non_flat_angles.append(tuple(p2.tolist()))
            non_flat_angles.insert(0, non_flat_angles[-1])
            non_flat_angles.pop(-1)
            # print(non_flat_angles)
            count += 1
        else:
            pass
    assert count == 1, 'count ==' + str(count) + '    ' + train_file

    # 断言类型为6的只有一个
    assert simple_cycles_semantics_train.count(6) == 1, train_file


    list1 = train_graph['corners']
    list2 = non_flat_angles
    if len(list1) >= 53:
        boundary_num.append(train_graph['file_id'])
    indices = [i for i, item in enumerate(list1) if item in list2]
    # print(list1)
    # print(list2)
    # print(indices)
    # print(train_graph['corners_np'])
    # print(train_graph['corner_list_np_normalized_padding_withsemantics'])


    boundary_vertex_indices_mask = np.zeros((53, 2), dtype=np.float32)
    for index in indices:
        boundary_vertex_indices_mask[index, :] = 1

    train_graph['boundary_vertex_indices'] = boundary_vertex_indices_mask
    # print(train_graph['boundary_vertex_indices'])

    '''【用于61的边界自注意力作为边界条件的边界邻接矩阵】和【用于61和60的CVAE（62、63）的边界坐标序列】'''

    # 边界邻接矩阵
    boundary_adjacency_matrix = np.zeros_like(train_graph['adjacency_matrix_np_padding'])
    for i, coord in enumerate(list2):
        list1_i = list1.index(coord)
        list1_i_plus_1 = list1.index(list2[(i + 1) % len(list2)])
        boundary_adjacency_matrix[list1_i, list1_i_plus_1] = 1
        boundary_adjacency_matrix[list1_i_plus_1, list1_i] = 1

    # print(boundary_adjacency_matrix)
    # print(list2)
    # assert 0


    # 用于61和60的CVAE（62、63）的边界坐标序列
    train_graph['boundary_vertex_coords_4cvae'] = list2
    train_graph['boundary_adjacency_matrix'] = boundary_adjacency_matrix




    # np.save(os.path.join('/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-withsemantics-withboundary-v2', f"{train_graph['file_id']}.npy"), train_graph)

    # v1 = copy.deepcopy(train_graph)
    # del v1['boundary_vertex_coords_4cvae']
    # del v1['boundary_adjacency_matrix']
    # np.save(os.path.join(
    #     '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-withsemantics-withboundary',
    #     f"{v1['file_id']}.npy"), v1)

print(boundary_num)