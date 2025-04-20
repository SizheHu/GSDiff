import os
from torch.utils.data import Dataset
import torch
import numpy as np
from datasets import tiny_graph

torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)

''' 房间类型是7类，实际不包含外部所以只有6类
 房间数最大是8最小是4'''
class RPlanGBubbleDiagram(Dataset):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        '''train(65763) & val(3000) & test(3000)'''
        if self.mode == 'train':
            self.files = os.listdir('../datasets/rplang-v3-bubble-diagram/train')
            self.randomize_data = True
        elif self.mode == 'val':
            self.files = os.listdir('../datasets/rplang-v3-bubble-diagram/val')
            self.randomize_data = False
        elif self.mode == 'test':
            self.files = os.listdir('../datasets/rplang-v3-bubble-diagram/test')
            self.randomize_data = False
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

        semantics = np.eye(7)[bbdiagram['semantics'] + [0] * (8 - len(bbdiagram['semantics']))].astype(np.float32)
        # 设计一套规则让每个节点都有一定概率转变为其他类型，边同理。
        # 我们统计了数据集中每一类节点的类型比例和边的0、1比例，把每一个节点（行向量）乘以以数据集节点分布复制行向量定义的转移矩阵，使得它的下一步满足数据集统计分布。
        # {0: 0.1512, 1: 0.3833, 2: 0.0071, 3: 0.1403, 4: 0.1603, 5: 0.1578, 6: 0}
        # {0: 0.6489, 1: 0.3511}
        if self.randomize_data:
            q_seman = np.array([[0.1512, 0.3833, 0.0071, 0.1403, 0.1603, 0.1578, 0],
                                [0.1512, 0.3833, 0.0071, 0.1403, 0.1603, 0.1578, 0],
                                [0.1512, 0.3833, 0.0071, 0.1403, 0.1603, 0.1578, 0],
                                [0.1512, 0.3833, 0.0071, 0.1403, 0.1603, 0.1578, 0],
                                [0.1512, 0.3833, 0.0071, 0.1403, 0.1603, 0.1578, 0],
                                [0.1512, 0.3833, 0.0071, 0.1403, 0.1603, 0.1578, 0],
                                [0.1512, 0.3833, 0.0071, 0.1403, 0.1603, 0.1578, 0],])
            semantics_augmented_distribution = np.dot(semantics, q_seman)
            semantics_augmented = np.eye(7)[np.array([np.random.choice(np.arange(7), p=d) for d in semantics_augmented_distribution])]
            semantics = semantics_augmented
        semantics[len(bbdiagram['semantics']):] = 0

        adjacency_matrix_ori = np.array(bbdiagram['adjacency_matrix'])
        if self.randomize_data:
            upper_indices = np.triu_indices(len(adjacency_matrix_ori), 0)
            upper_tri_flatten = adjacency_matrix_ori[upper_indices]
            upper_tri_onehot = np.eye(2)[upper_tri_flatten]
            q_edge = np.array([[0.6489, 0.3511],
                               [0.6489, 0.3511]])
            edge_augmented_distribution = np.dot(upper_tri_onehot, q_edge)
            edge_augmented = np.eye(2)[np.array([np.random.choice(np.arange(2), p=d) for d in edge_augmented_distribution])]
            upper_tri_onehot_augmented = edge_augmented[:, 1]
            edges = np.zeros_like(adjacency_matrix_ori)
            edges[np.triu_indices(len(adjacency_matrix_ori), 0)] = upper_tri_onehot_augmented
            adjacency_matrix_ori = edges + edges.T - np.diag(edges.diagonal())

        adjacency_matrix = np.zeros((8, 8), dtype=np.uint8)
        adjacency_matrix[:len(bbdiagram['semantics']), :len(bbdiagram['semantics'])] = adjacency_matrix_ori


        semantics_padding_mask = np.zeros((8, 1), dtype=np.uint8)
        semantics_padding_mask[:len(bbdiagram['semantics']), :] = 1

        room_number = np.zeros((1, 5), dtype=np.uint8)
        room_number[0, len(bbdiagram['semantics']) - 4] = 1

        global_matrix = np.zeros((8, 8), dtype=np.uint8)
        global_matrix[:len(bbdiagram['semantics']), :len(bbdiagram['semantics'])] = 1



        return semantics, adjacency_matrix, semantics_padding_mask, global_matrix, room_number