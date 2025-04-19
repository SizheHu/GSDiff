import os
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from PIL import Image, ImageDraw
from tqdm import tqdm

torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)


class RPlanGEdgeSemanSimplified_81(Dataset):
    def __init__(self, mode): # 不随机数据
        super().__init__()
        self.mode = mode
        '''train(65763) & val(3000) & test(3000)'''
        if self.mode == 'train':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics-withboundary/train')
        elif self.mode == 'val':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics-withboundary/val')
        elif self.mode == 'test':
            self.files = os.listdir('../datasets/rplang-v3-withsemantics-withboundary/test')
        else:
            assert 0, 'mode error'
        self.files = sorted(self.files, key=lambda x: int(x[:-4]), reverse=False)
        self.ftmps = []
        for fn in tqdm(self.files):
            self.ftmps.append(np.load('../datasets/prerunning_cnn_featuremaps/' + fn, allow_pickle=True).item()[16][0])


    def __len__(self):
        '''return len(dataset)'''
        return len(self.files)

    def __getitem__(self, index):

        # if self.mode == 'train':
        #     feat = np.load('../datasets/prerunning_cnn_featuremaps/' + self.files[index], allow_pickle=True).item()
        #     # featmap_64 = feat[64][0] # ndarray(256, 64, 64)
        #     # featmap_32 = feat[32][0] # ndarray(512, 32, 32)
        #     featmap_16 = feat[16][0] # ndarray(1024, 16, 16)
        # elif self.mode == 'val':
        #     feat = np.load('../datasets/prerunning_cnn_featuremaps/' + self.files[index], allow_pickle=True).item()
        #     # featmap_64 = feat[64][0] # ndarray(256, 64, 64)
        #     # featmap_32 = feat[32][0] # ndarray(512, 32, 32)
        #     featmap_16 = feat[16][0] # ndarray(1024, 16, 16)
        # elif self.mode == 'test':
        #     feat = np.load('../datasets/prerunning_cnn_featuremaps/' + self.files[index], allow_pickle=True).item()
        #     # featmap_64 = feat[64][0] # ndarray(256, 64, 64)
        #     # featmap_32 = feat[32][0] # ndarray(512, 32, 32)
        #     featmap_16 = feat[16][0] # ndarray(1024, 16, 16)
        # else:
        #     assert 0




    

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
        # global_attn_matrix = structure_graph['global_matrix_np_padding'].astype(np.uint8)
        global_attn_matrix = np.ones((53, 53), dtype=np.uint8)
        '''corners padding mask, (53, 1)'''
        corners_padding_mask = graph['padding_mask'] # uint8

        featmap_16 = self.ftmps[index]
 



        

        # return featmap_64, featmap_32, featmap_16, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask
        # return featmap_32, featmap_16, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask
        return featmap_16, corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask



