import copy
import math
import os, json, numpy as np, cv2, random
from collections import Counter
import networkx as nx
from tqdm import *

image_path = r'/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/rplan_dataset/floorplan_dataset'
# print(len(os.listdir(image_path)))

train = [int(fn[:-4]) for fn in os.listdir(r'/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-withsemantics/train')]
test = [int(fn[:-4]) for fn in os.listdir(r'/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-withsemantics/test')]


for sample_id in tqdm(range(0, 80788)):
    '''read corners annotation'''
    if sample_id in train:
        graph = np.load('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-withsemantics/train/' + str(sample_id) + '.npy', allow_pickle=True).item()
    elif sample_id in test:
        graph = np.load('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-withsemantics/test/' + str(sample_id) + '.npy', allow_pickle=True).item()
    else:
        continue
    # 对'semantics'键的值进行处理
    semantics = graph['semantics']
    corner_list = graph['corner_list_np_normalized_padding']
    new_semantics = {}
    for key, value in semantics.items():
        # 修改二维坐标元组为原来的1/2
        new_key = ((key[0] - 128) / 128, (key[1] - 128) / 128)
        new_semantics[new_key] = value
    # 替换'semantics'的值
    graph['semantics'] = new_semantics
    # print(graph)

    # print(graph['semantics'])
    # print(graph['corner_list_np_normalized_padding'])

    # 创建一个新的空数组，尺寸为(53, 16)
    result = np.zeros((53, 16), dtype=corner_list.dtype)
    # 遍历'corner_list'数组
    for idx, coord in enumerate(corner_list):
        # 将坐标元组转换为原始精度
        coord_tuple = (coord[0], coord[1])
        # print(coord_tuple)
        # print(new_semantics)
        # 检查坐标是否在'semantics'字典中
        if coord_tuple in new_semantics:
            vector = new_semantics[coord_tuple]
        else:
            if idx < len(new_semantics):
                assert 0
            vector = [0] * 14  # 如果不在字典中，创建一个全0向量
        # 将'corner_list'中的二维元素与'semantics'中的值拼接
        result[idx] = np.concatenate((coord, vector))
    # print(result)

    graph['corner_list_np_normalized_padding_withsemantics'] = result
    # assert 0
    #
    if sample_id in train:
        np.save('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-withsemantics/train/' + str(sample_id) + '.npy', graph)
    elif sample_id in test:
        np.save('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-withsemantics/test/' + str(sample_id) + '.npy', graph)

