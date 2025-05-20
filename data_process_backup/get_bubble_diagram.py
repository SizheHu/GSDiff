import sys
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
bubblenumber = {}
val_files = os.listdir('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v3-withsemantics/test')
for val_file in tqdm(val_files):
    val_graph = np.load('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v3-withsemantics/test/' + val_file, allow_pickle=True).item()

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
    # # 初始化一个n*9的新数组(53, 9)
    # corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
    # # 复制第0、1列 坐标
    # corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
    # # 计算新的第2列 标签0 客厅/餐厅/玄关
    # corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6]]).sum(axis=1)
    # # 计算新的第3列 标签1 卧室/书房
    # corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [4, 5, 6, 9, 10]]).sum(axis=1)
    # # 计算新的第4列 标签2 橱柜
    # corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
    # # 其他
    # corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 7]  # 标签3 厨房
    # corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 8]  # 标签4 卫生间
    # corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]  # 标签5 阳台
    # corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 17]  # 标签6 外部

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


    # 存储气泡图，我们可以先存储更丰富的信息备用。保存气泡图的房间多边形（首尾相同）、重心、类型、邻接矩阵。
    bbdiagram = {}
    bbdiagram['file_id'] = val_graph['file_id']
    bbdiagram['polygons'] = simple_cycles_val
    bbdiagram['centroids'] = centroids
    bbdiagram['semantics'] = simple_cycles_semantics_val
    for s in simple_cycles_semantics_val:
        bubblecategory[s] += 1

    bbdiagram['adjacency_matrix'] = adjacency_matrix
    bbdiagram['corner_number'] = len(val_graph['corners'])
    # print(bbdiagram['corner_number'])
    # np.save(os.path.join('/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-bubble-diagram', f"{val_graph['file_id']}.npy"), bbdiagram)
    # bubblenumber.append(len(bbdiagram['polygons']))
    bubblenumber[bbdiagram['file_id']] = len(bbdiagram['polygons'])

    # 去除房间数大于8的 (31554 -> 31504)
    # if len(bbdiagram['polygons']) > 8:
    #     os.remove(os.path.join('/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-bubble-diagram', f"{val_graph['file_id']}.npy"))
    #     os.remove(os.path.join(
    #         '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-withsemantics-withboundary',
    #         f"{val_graph['file_id']}.npy"))
    #     os.remove(os.path.join(
    #         '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-withsemantics',
    #         f"{val_graph['file_id']}.npy"))
    #     os.remove(os.path.join(
    #         '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-withsemantics-withboundary-v2',
    #         f"{val_graph['file_id']}.npy"))
    # 此后进行两次划分：房间数等于8的样本单独复制一份用于HouseDiffusion的比较；随机取出3000个样本用于其他对比
    # os.mkdir(
    #     '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_8/')
    # os.mkdir('/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_8/rplang-v3-bubble-diagram/')
    # os.mkdir(
    #     '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_8/rplang-v3-withsemantics-withboundary/')
    # os.mkdir(
    #     '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_8/rplang-v3-withsemantics/')
    # os.mkdir(
    #     '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_8/rplang-v3-withsemantics-withboundary-v2/')
    # if len(bbdiagram['polygons']) == 8:
    #     shutil.copy(os.path.join('/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-bubble-diagram', f"{val_graph['file_id']}.npy"),
    #                 '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_8/rplang-v3-bubble-diagram/')
    #     shutil.copy(os.path.join(
    #         '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-withsemantics-withboundary',
    #         f"{val_graph['file_id']}.npy"),
    #                 '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_8/rplang-v3-withsemantics-withboundary/')
    #     shutil.copy(os.path.join(
    #         '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-withsemantics',
    #         f"{val_graph['file_id']}.npy"),
    #                 '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_8/rplang-v3-withsemantics/')
    #     shutil.copy(os.path.join(
    #         '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_3w/rplang-v3-withsemantics-withboundary-v2',
    #         f"{val_graph['file_id']}.npy"),
    #                 '/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/test_3w/Data/test_8/rplang-v3-withsemantics-withboundary-v2/')





# train_files = os.listdir('../datasets/rplang-v3-withsemantics')
# for train_file in tqdm(train_files):
#     train_graph = np.load('../datasets/rplang-v3-withsemantics/train/' + train_file, allow_pickle=True).item()
#
#     '''file_id
#         corners
#         adjacency_matrix
#         adjacency_list
#         corners_np
#         adjacency_matrix_np
#         adjacency_list_np
#         corner_list_np_normalized
#         corner_list_np_normalized_padding
#         padding_mask
#         global_matrix_np_padding
#         adjacency_matrix_np_padding
#         edge_coords
#         edges
#         semantics
#         corner_list_np_normalized_padding_withsemantics
#         '''
#     '''coords_withsemantics, (53, 16)'''
#     corners_withsemantics = train_graph['corner_list_np_normalized_padding_withsemantics']
#     # 初始化一个n*9的新数组(53, 9)
#     corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
#     # 复制第0、1列
#     corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
#     # 计算新的第2列
#     corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
#     # 计算新的第3列
#     corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
#     # 计算新的第4列
#     corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
#     # 复制第4、5、11、15列
#     corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4]
#     corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5]
#     corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]
#     corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15]
#
#     '''attn 1 matrix, (53, 53)'''
#     global_attn_matrix = train_graph['global_matrix_np_padding'].astype(bool)
#     '''corners padding mask, (53, 1)'''
#     corners_padding_mask = train_graph['padding_mask']
#
#     '''edges, (2809, 1)'''
#     edges = train_graph['edges']
#     corners_withsemantics_0_train = corners_withsemantics_simplified[None, :, :]
#     global_attn_matrix_train = global_attn_matrix[None, :, :]
#     corners_padding_mask_train = corners_padding_mask[None, :, :]
#     edges_train = edges[None, :, :]
#     corners_withsemantics_0_train = corners_withsemantics_0_train.clip(-1, 1)
#     corners_0_train = (corners_withsemantics_0_train[0, :, :2] * 128 + 128).astype(int)
#     semantics_0_train = corners_withsemantics_0_train[0, :, 2:].astype(int)
#     global_attn_matrix_train = global_attn_matrix_train
#     corners_padding_mask_train = corners_padding_mask_train
#     edges_train = edges_train
#     corners_0_train_depadded = corners_0_train[corners_padding_mask_train.squeeze() == 1][None, :, :]  # (n, 2)
#     semantics_0_train_depadded = semantics_0_train[corners_padding_mask_train.squeeze() == 1][None, :, :]  # (n, 7)
#     edges_train_depadded = edges_train[global_attn_matrix_train.reshape(1, -1, 1)][None, :, None]
#     edges_train_depadded = np.concatenate((1 - edges_train_depadded, edges_train_depadded), axis=2)
#
#     ''' get planar cycles'''
#     # 形状为 (1, n, 14) 的 ndarray，包含 0 和 1;找到每个子数组中 1 所在的索引,用 99999 替换值为 0 的原始元素
#     semantics_gt_i_transform_train = semantics_0_train_depadded
#     semantics_gt_i_transform_indices_train = np.indices(semantics_gt_i_transform_train.shape)[-1]
#     semantics_gt_i_transform_train = np.where(semantics_gt_i_transform_train == 1,
#                                               semantics_gt_i_transform_indices_train, 99999)
#
#     gt_i_points_train = [tuple(corner_with_seman_train) for corner_with_seman_train in
#                          np.concatenate((corners_0_train_depadded, semantics_gt_i_transform_train), axis=-1).tolist()[
#                              0]]
#     # print(output_points)
#     gt_i_edges_train = edges_to_coordinates(
#         np.triu(edges_train_depadded[0, :, 1].reshape(len(gt_i_points_train), len(gt_i_points_train))).reshape(-1),
#         gt_i_points_train)
#
#     # print(gt_i_points_train)
#     # print(gt_i_edges_train)
#
#     d_rev_train, simple_cycles_train, simple_cycles_semantics_train = get_cycle_basis_and_semantic_2_semansimplified(
#         gt_i_points_train,
#         gt_i_edges_train)
#     simple_cycles_train_ = []
#     for sc in simple_cycles_train:
#         sc_train = [(t[0], t[1]) for t in sc]
#         simple_cycles_train_.append(sc_train)
#         # print(sc_train)
#     # for scs in simple_cycles_semantics_train:
#     #     print(scs)
#     polygons = simple_cycles_train_
#     edges = [[(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))][:-1] for polygon in polygons]
#     # print(edges)
#
#
#     def get_adjacency_matrix(polygons):
#         n = len(polygons)
#         matrix = [[0] * n for _ in range(n)]
#         for i in range(n):
#             for j in range(i + 1, n):
#                 if any(set(edge) in [set(edge_j) for edge_j in polygons[j]] for edge in polygons[i]):
#                     matrix[i][j] = 1
#                     matrix[j][i] = 1
#         return matrix
#
#
#     adjacency_matrix = get_adjacency_matrix(edges)
#     edgecategory[1] += np.sum(np.triu(np.array(adjacency_matrix)))
#     edgecategory[0] += ((len(adjacency_matrix) * (len(adjacency_matrix) + 1)) / 2) - np.sum(np.triu(np.array(adjacency_matrix)))
#
#
#     # 使用Shoelace公式来计算面积，然后使用重心的公式来计算凹多边形的重心。在考虑邻接性并绘制时，我们可以使用OpenCV图形库。以下是相关的Python代码：
#     # 计算多边形的重心
#     def get_polygon_centroid(polygon):
#         area = 0
#         x = 0
#         y = 0
#         for i in range(-1, len(polygon) - 1):
#             step = (polygon[i][0] * polygon[i + 1][1]) - (polygon[i + 1][0] * polygon[i][1])
#             area += step
#             x += (polygon[i][0] + polygon[i + 1][0]) * step
#             y += (polygon[i][1] + polygon[i + 1][1]) * step
#         area /= 2
#         x /= (6 * area)
#         y /= (6 * area)
#         return (int(x), int(y))
#
#
#
#
#     # 计算每个多边形的重心
#     centroids = [get_polygon_centroid(polygon[:-1]) for polygon in polygons]
#
#
#     # 存储气泡图，我们可以先存储更丰富的信息备用。保存气泡图的房间多边形（首尾相同）、重心、类型、邻接矩阵。
#     bbdiagram = {}
#     bbdiagram['file_id'] = train_graph['file_id']
#     bbdiagram['polygons'] = simple_cycles_train
#     bbdiagram['centroids'] = centroids
#     bbdiagram['semantics'] = simple_cycles_semantics_train
#     for s in simple_cycles_semantics_train:
#         bubblecategory[s] += 1
#
#     bbdiagram['adjacency_matrix'] = adjacency_matrix
#     bbdiagram['corner_number'] = len(train_graph['corners'])
#     # print(bbdiagram['corner_number'])
#     # np.save(os.path.join('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v3-bubble-diagram/train', f"{train_graph['file_id']}.npy"), bbdiagram)
#     # bubblenumber.append(len(bbdiagram['polygons']))
#
#
#
#
# test_files = os.listdir('../datasets/rplang-v3-withsemantics/test')
# for test_file in tqdm(test_files):
#     test_graph = np.load('../datasets/rplang-v3-withsemantics/test/' + test_file, allow_pickle=True).item()
#
#     '''file_id
#         corners
#         adjacency_matrix
#         adjacency_list
#         corners_np
#         adjacency_matrix_np
#         adjacency_list_np
#         corner_list_np_normalized
#         corner_list_np_normalized_padding
#         padding_mask
#         global_matrix_np_padding
#         adjacency_matrix_np_padding
#         edge_coords
#         edges
#         semantics
#         corner_list_np_normalized_padding_withsemantics
#         '''
#     '''coords_withsemantics, (53, 16)'''
#     corners_withsemantics = test_graph['corner_list_np_normalized_padding_withsemantics']
#     # 初始化一个n*9的新数组(53, 9)
#     corners_withsemantics_simplified = np.zeros((corners_withsemantics.shape[0], 9))
#     # 复制第0、1列
#     corners_withsemantics_simplified[:, 0:2] = corners_withsemantics[:, 0:2]
#     # 计算新的第2列
#     corners_withsemantics_simplified[:, 2] = (corners_withsemantics[:, [2, 6, 12]]).sum(axis=1)
#     # 计算新的第3列
#     corners_withsemantics_simplified[:, 3] = (corners_withsemantics[:, [3, 7, 8, 9, 10]]).sum(axis=1)
#     # 计算新的第4列
#     corners_withsemantics_simplified[:, 4] = (corners_withsemantics[:, [13, 14]]).sum(axis=1)
#     # 复制第4、5、11、15列
#     corners_withsemantics_simplified[:, 5] = corners_withsemantics[:, 4]
#     corners_withsemantics_simplified[:, 6] = corners_withsemantics[:, 5]
#     corners_withsemantics_simplified[:, 7] = corners_withsemantics[:, 11]
#     corners_withsemantics_simplified[:, 8] = corners_withsemantics[:, 15]
#
#     '''attn 1 matrix, (53, 53)'''
#     global_attn_matrix = test_graph['global_matrix_np_padding'].astype(bool)
#     '''corners padding mask, (53, 1)'''
#     corners_padding_mask = test_graph['padding_mask']
#
#     '''edges, (2809, 1)'''
#     edges = test_graph['edges']
#     corners_withsemantics_0_test = corners_withsemantics_simplified[None, :, :]
#     global_attn_matrix_test = global_attn_matrix[None, :, :]
#     corners_padding_mask_test = corners_padding_mask[None, :, :]
#     edges_test = edges[None, :, :]
#     corners_withsemantics_0_test = corners_withsemantics_0_test.clip(-1, 1)
#     corners_0_test = (corners_withsemantics_0_test[0, :, :2] * 128 + 128).astype(int)
#     semantics_0_test = corners_withsemantics_0_test[0, :, 2:].astype(int)
#     global_attn_matrix_test = global_attn_matrix_test
#     corners_padding_mask_test = corners_padding_mask_test
#     edges_test = edges_test
#     corners_0_test_depadded = corners_0_test[corners_padding_mask_test.squeeze() == 1][None, :, :]  # (n, 2)
#     semantics_0_test_depadded = semantics_0_test[corners_padding_mask_test.squeeze() == 1][None, :, :]  # (n, 7)
#     edges_test_depadded = edges_test[global_attn_matrix_test.reshape(1, -1, 1)][None, :, None]
#     edges_test_depadded = np.concatenate((1 - edges_test_depadded, edges_test_depadded), axis=2)
#
#     ''' get planar cycles'''
#     # 形状为 (1, n, 14) 的 ndarray，包含 0 和 1;找到每个子数组中 1 所在的索引,用 99999 替换值为 0 的原始元素
#     semantics_gt_i_transform_test = semantics_0_test_depadded
#     semantics_gt_i_transform_indices_test = np.indices(semantics_gt_i_transform_test.shape)[-1]
#     semantics_gt_i_transform_test = np.where(semantics_gt_i_transform_test == 1,
#                                               semantics_gt_i_transform_indices_test, 99999)
#
#     gt_i_points_test = [tuple(corner_with_seman_test) for corner_with_seman_test in
#                          np.concatenate((corners_0_test_depadded, semantics_gt_i_transform_test), axis=-1).tolist()[
#                              0]]
#     # print(output_points)
#     gt_i_edges_test = edges_to_coordinates(
#         np.triu(edges_test_depadded[0, :, 1].reshape(len(gt_i_points_test), len(gt_i_points_test))).reshape(-1),
#         gt_i_points_test)
#
#     # print(gt_i_points_test)
#     # print(gt_i_edges_test)
#
#     d_rev_test, simple_cycles_test, simple_cycles_semantics_test = get_cycle_basis_and_semantic_2_semansimplified(
#         gt_i_points_test,
#         gt_i_edges_test)
#     simple_cycles_test_ = []
#     for sc in simple_cycles_test:
#         sc_test = [(t[0], t[1]) for t in sc]
#         simple_cycles_test_.append(sc_test)
#         # print(sc_test)
#     # for scs in simple_cycles_semantics_test:
#     #     print(scs)
#     polygons = simple_cycles_test_
#     edges = [[(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))][:-1] for polygon in polygons]
#     # print(edges)
#
#
#     def get_adjacency_matrix(polygons):
#         n = len(polygons)
#         matrix = [[0] * n for _ in range(n)]
#         for i in range(n):
#             for j in range(i + 1, n):
#                 if any(set(edge) in [set(edge_j) for edge_j in polygons[j]] for edge in polygons[i]):
#                     matrix[i][j] = 1
#                     matrix[j][i] = 1
#         return matrix
#
#
#     adjacency_matrix = get_adjacency_matrix(edges)
#     edgecategory[1] += np.sum(np.triu(np.array(adjacency_matrix)))
#     edgecategory[0] += ((len(adjacency_matrix) * (len(adjacency_matrix) + 1)) / 2) - np.sum(np.triu(np.array(adjacency_matrix)))
#
#
#     # 使用Shoelace公式来计算面积，然后使用重心的公式来计算凹多边形的重心。在考虑邻接性并绘制时，我们可以使用OpenCV图形库。以下是相关的Python代码：
#     # 计算多边形的重心
#     def get_polygon_centroid(polygon):
#         area = 0
#         x = 0
#         y = 0
#         for i in range(-1, len(polygon) - 1):
#             step = (polygon[i][0] * polygon[i + 1][1]) - (polygon[i + 1][0] * polygon[i][1])
#             area += step
#             x += (polygon[i][0] + polygon[i + 1][0]) * step
#             y += (polygon[i][1] + polygon[i + 1][1]) * step
#         area /= 2
#         x /= (6 * area)
#         y /= (6 * area)
#         return (int(x), int(y))
#
#
#
#
#     # 计算每个多边形的重心
#     centroids = [get_polygon_centroid(polygon[:-1]) for polygon in polygons]
#
#
#     # 存储气泡图，我们可以先存储更丰富的信息备用。保存气泡图的房间多边形（首尾相同）、重心、类型、邻接矩阵。
#     bbdiagram = {}
#     bbdiagram['file_id'] = test_graph['file_id']
#     bbdiagram['polygons'] = simple_cycles_test
#     bbdiagram['centroids'] = centroids
#     bbdiagram['semantics'] = simple_cycles_semantics_test
#     for s in simple_cycles_semantics_test:
#         bubblecategory[s] += 1
#
#     bbdiagram['adjacency_matrix'] = adjacency_matrix
#     bbdiagram['corner_number'] = len(test_graph['corners'])
#     # print(bbdiagram['corner_number'])
#     # np.save(os.path.join('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v3-bubble-diagram/test', f"{test_graph['file_id']}.npy"), bbdiagram)
#     # bubblenumber.append(len(bbdiagram['polygons']))

# a = {0: 73516, 1: 186380, 2: 3457, 3: 68240, 4: 77937, 5: 76701}
# b = {0: 1246027.0, 1: 674165}
# asum = sum([a[i] for i in a.keys()])
# bsum = sum([b[i] for i in b.keys()])
# a_ = {}
# b_ = {}
# for k, v in a.items():
#     a_[k] = v / asum
# for k, v in b.items():
#     b_[k] = v / bsum
# print(a_)
# print(b_)

# m = 9990
# test_files = os.listdir('../datasets/rplang-v3-bubble-diagram/test')
# for test_file in tqdm(test_files):
#     test_graph = np.load('../datasets/rplang-v3-bubble-diagram/test/' + test_file, allow_pickle=True).item()
#     if test_graph['corner_number'] < m:
#         m = test_graph['corner_number']
# print(m)


b8 = []
for k, v in bubblenumber.items():
    if v == 8:
        b8.append(k)
print(b8)