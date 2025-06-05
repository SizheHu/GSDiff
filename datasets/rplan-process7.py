import sys

import numpy as np

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
import copy
import os
from tiny_graph import test as test_tiny_graph


test_fnids = [int(fnid[:-4]) for fnid in test_tiny_graph]


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
    subdirs = ['test']
    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        file_list = [f for f in os.listdir(full_path)
                     if os.path.isfile(os.path.join(full_path, f))]
        print('文件数' + str(len(file_list)))


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

if not os.path.exists('./rplandata/Data/rplang-v3-withsemantics-withboundary'):
    os.mkdir('./rplandata/Data/rplang-v3-withsemantics-withboundary')
os.mkdir('./rplandata/Data/rplang-v3-withsemantics-withboundary/test')

if not os.path.exists('./rplandata/Data/rplang-v3-withsemantics-withboundary-v2'):
    os.mkdir('./rplandata/Data/rplang-v3-withsemantics-withboundary-v2')
os.mkdir('./rplandata/Data/rplang-v3-withsemantics-withboundary-v2/test')

test_path = './rplandata/Data/rplang-v3-withsemantics/test'

test_files = os.listdir(test_path)
for test_file in tqdm(test_files):
    # print(test_file)
    test_graph = np.load(test_path + '/' + test_file, allow_pickle=True).item()

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
    corners_withsemantics = test_graph['corner_list_np_normalized_padding_withsemantics']
    # 初始化一个n*9的新数组(53, 9)
    corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
    # 复制第0、1列 坐标
    corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
    # 计算新的第2列 标签0 客厅/餐厅/玄关
    corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
    # 计算新的第3列 标签1 卧室/书房
    corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
    # 计算新的第4列 标签2 橱柜
    corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
    # 其他
    corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4] # 标签3 厨房
    corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5] # 标签4 卫生间
    corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11] # 标签5 阳台
    corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15] # 标签6 外部

    '''attn 1 matrix, (53, 53)'''
    global_attn_matrix = test_graph['global_matrix_np_padding'].astype(bool)
    '''corners padding mask, (53, 1)'''
    corners_padding_mask = test_graph['padding_mask']

    '''edges, (2809, 1)'''
    edges = test_graph['edges']
    corners_withsemantics_0_test = corners_withsemantics_simplified[None, :, :]
    global_attn_matrix_test = global_attn_matrix[None, :, :]
    corners_padding_mask_test = corners_padding_mask[None, :, :]
    edges_test = edges[None, :, :]
    corners_withsemantics_0_test = corners_withsemantics_0_test.clip(-1, 1)
    corners_0_test = (corners_withsemantics_0_test[0, :, :2] * 128 + 128).astype(int)
    semantics_0_test = corners_withsemantics_0_test[0, :, 2:].astype(int)
    global_attn_matrix_test = global_attn_matrix_test
    corners_padding_mask_test = corners_padding_mask_test
    edges_test = edges_test
    corners_0_test_depadded = corners_0_test[corners_padding_mask_test.squeeze() == 1][None, :, :]  # (n, 2)
    semantics_0_test_depadded = semantics_0_test[corners_padding_mask_test.squeeze() == 1][None, :, :]  # (n, 7)
    edges_test_depadded = edges_test[global_attn_matrix_test.reshape(1, -1, 1)][None, :, None]
    edges_test_depadded = np.concatenate((1 - edges_test_depadded, edges_test_depadded), axis=2)

    ''' get planar cycles'''
    # 形状为 (1, n, 14) 的 ndarray，包含 0 和 1;找到每个子数组中 1 所在的索引,用 99999 替换值为 0 的原始元素
    semantics_gt_i_transform_test = semantics_0_test_depadded
    semantics_gt_i_transform_indices_test = np.indices(semantics_gt_i_transform_test.shape)[-1]
    semantics_gt_i_transform_test = np.where(semantics_gt_i_transform_test == 1,
                                              semantics_gt_i_transform_indices_test, 99999)

    gt_i_points_test = [tuple(corner_with_seman_test) for corner_with_seman_test in
                         np.concatenate((corners_0_test_depadded, semantics_gt_i_transform_test), axis=-1).tolist()[
                             0]]
    # print(output_points)
    gt_i_edges_test = edges_to_coordinates(
        np.triu(edges_test_depadded[0, :, 1].reshape(len(gt_i_points_test), len(gt_i_points_test))).reshape(-1),
        gt_i_points_test)

    # print(gt_i_points_test)
    # print(gt_i_edges_test)

    d_rev_test, simple_cycles_test, simple_cycles_semantics_test = get_cycle_basis_and_semantic_2_semansimplified_4extractingboundary(
        gt_i_points_test,
        gt_i_edges_test)
    simple_cycles_test_ = []
    for sc in simple_cycles_test:
        sc_test = [(t[0], t[1]) for t in sc]
        simple_cycles_test_.append(sc_test)
        # print(sc_test)
    # for scs in simple_cycles_semantics_test:
    #     print(scs)
    polygons = simple_cycles_test_
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
        if simple_cycles_semantics_test[polygon_i] == 6:
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
    assert count == 1, 'count ==' + str(count) + '    ' + test_file

    # 断言类型为6的只有一个
    assert simple_cycles_semantics_test.count(6) == 1, test_file


    list1 = test_graph['corners']
    list2 = non_flat_angles
    if len(list1) >= 53:
        boundary_num.append(test_graph['file_id'])
    indices = [i for i, item in enumerate(list1) if item in list2]
    # print(list1)
    # print(list2)
    # print(indices)
    # print(test_graph['corners_np'])
    # print(test_graph['corner_list_np_normalized_padding_withsemantics'])


    boundary_vertex_indices_mask = np.zeros((53, 2), dtype=np.float32)
    for index in indices:
        boundary_vertex_indices_mask[index, :] = 1

    test_graph['boundary_vertex_indices'] = boundary_vertex_indices_mask
    # print(test_graph['boundary_vertex_indices'])

    # 边界邻接矩阵
    boundary_adjacency_matrix = np.zeros_like(test_graph['adjacency_matrix_np_padding'])
    for i, coord in enumerate(list2):
        list1_i = list1.index(coord)
        list1_i_plus_1 = list1.index(list2[(i + 1) % len(list2)])
        boundary_adjacency_matrix[list1_i, list1_i_plus_1] = 1
        boundary_adjacency_matrix[list1_i_plus_1, list1_i] = 1

    # print(boundary_adjacency_matrix)
    # print(list2)
    # assert 0


    # 边界坐标序列
    test_graph['boundary_vertex_coords_4cvae'] = list2
    test_graph['boundary_adjacency_matrix'] = boundary_adjacency_matrix



    v1 = copy.deepcopy(test_graph)
    del v1['boundary_vertex_coords_4cvae']
    del v1['boundary_adjacency_matrix']

    # 检验
    gt1 = np.load('../../house_diffusion-main/datasets/rplang-v3-withsemantics-withboundary/test/' + f"{test_graph['file_id']}.npy", allow_pickle=True).item()
    gt2 = np.load('../../house_diffusion-main/datasets/rplang-v3-withsemantics-withboundary-v2/test/' + f"{v1['file_id']}.npy", allow_pickle=True).item()
    assert deep_compare(test_graph, gt2), str(test_graph['file_id'])
    assert deep_compare(v1, gt1), str(test_graph['file_id'])

    np.save(os.path.join('rplandata/Data/rplang-v3-withsemantics-withboundary-v2/test', f"{test_graph['file_id']}.npy"), test_graph)
    np.save(os.path.join('rplandata/Data/rplang-v3-withsemantics-withboundary/test',f"{v1['file_id']}.npy"), v1)





# 检查子目录文件的数量
check_subdir_file_counts('./rplandata/Data/rplang-v3-withsemantics')
check_subdir_file_counts('./rplandata/Data/rplang-v3-withsemantics-withboundary')
check_subdir_file_counts('./rplandata/Data/rplang-v3-withsemantics-withboundary-v2')