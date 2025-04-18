import os
from torch.utils.data import Dataset
import torch
import numpy as np
from datasets import tiny_graph

torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)


class RPlanGEdgeSemanSimplified_56_31(Dataset):
    '''
    hould have storaged data without padding, augmentation and normalization in advance.
   data is graphs, (V, E), V has attributes(coords, ...), E has adjacency matrices.
   (although we use ordered data structure like ndarray, we only use the order in adjacency matrices(instead of adjacency lists)
   to facilitate the data loading. we don't use the order in nn, to meet permutation invariability of graph nodes.)
    '''
    def __init__(self, mode):
        '''(1)data reading. np.load()
           (2)data filtering(generate_corner_number in eval and not in train).
           (3)normalization(you could also do this in __getitem__() even after batch loading, but this will lead to longer time in training and sampling).
           purpose: make nn do not need to learn distributions scale shifting, more easier to converge.
           (4)padding and attn mask generating.

           but if dataset very big, memory can't stand, you should storage each one as a file and read it'''
        super().__init__()
        self.mode = mode
        '''train(65763) & val(3000) & test(3000)'''
        if self.mode == 'train':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics/train')
        elif self.mode == 'val':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics/val')
        elif self.mode == 'test':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics/test')
        else:
            assert 0, 'mode error'
        self.files = sorted(self.files, key=lambda x: int(x[:-4]), reverse=False)

    def __len__(self):
        '''return len(dataset)'''
        return len(self.files)

    def __getitem__(self, index):
        if self.mode == 'train':
            bbdiagram = np.load('../datasets/rplang-v3-bubble-diagram/train/' + self.files[index], allow_pickle=True).item()
        elif self.mode == 'val':
            bbdiagram = np.load('../datasets/rplang-v3-bubble-diagram/val/' + self.files[index], allow_pickle=True).item()
        elif self.mode == 'test':
            bbdiagram = np.load('../datasets/rplang-v3-bubble-diagram/test/' + self.files[index], allow_pickle=True).item()
        else:
            assert 0, 'mode error'


        semantics = np.eye(7)[bbdiagram['semantics'] + [0] * (8 - len(bbdiagram['semantics']))].astype(np.float64)
        # 设计一套规则让每个节点都有一定概率转变为其他类型，边同理。
        # 我们统计了数据集中每一类节点的类型比例和边的0、1比例，把每一个节点（行向量）乘以以数据集节点分布复制行向量定义的转移矩阵，使得它的下一步满足数据集统计分布。
        # {0: 0.1512, 1: 0.3833, 2: 0.0071, 3: 0.1403, 4: 0.1603, 5: 0.1578, 6: 0}
        # {0: 0.6489, 1: 0.3511}
        semantics[len(bbdiagram['semantics']):] = 0



        adjacency_matrix_ori = np.array(bbdiagram['adjacency_matrix'])
        

        adjacency_matrix = np.zeros((8, 8), dtype=np.uint8)
        adjacency_matrix[:len(bbdiagram['semantics']), :len(bbdiagram['semantics'])] = adjacency_matrix_ori




        semantics_padding_mask = np.zeros((8, 1), dtype=np.uint8)
        semantics_padding_mask[:len(bbdiagram['semantics']), :] = 1

        room_number = np.zeros((1, 5), dtype=np.uint8)
        room_number[0, len(bbdiagram['semantics']) - 4] = 1

        global_matrix = np.zeros((8, 8), dtype=np.uint8)
        global_matrix[:len(bbdiagram['semantics']), :len(bbdiagram['semantics'])] = 1










        

        if self.mode == 'train':
            graph = np.load('../datasets/rplang-v3-withsemantics/train/' + self.files[index], allow_pickle=True).item()
        elif self.mode == 'val':
            graph = np.load('../datasets/rplang-v3-withsemantics/val/' + self.files[index], allow_pickle=True).item()
        elif self.mode == 'test':
            graph = np.load('../datasets/rplang-v3-withsemantics/test/' + self.files[index], allow_pickle=True).item()
        else:
            assert 0, 'mode error'

        '''coords_withsemantics, (53, 16)'''
        corners_withsemantics = graph['corner_list_np_normalized_padding_withsemantics']
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
        global_attn_matrix = graph['global_matrix_np_padding'].astype(bool)
        '''corners padding mask, (53, 1)'''
        corners_padding_mask = graph['padding_mask']

        '''edges, (2809, 1)'''
        edges = graph['edges']











        
        return semantics, adjacency_matrix, semantics_padding_mask, global_matrix, room_number, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask, edges