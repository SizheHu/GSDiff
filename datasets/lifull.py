import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
torch.set_printoptions(threshold=np.inf, linewidth=999999)
np.set_printoptions(threshold=np.inf, linewidth=999999)



 
class lifull(Dataset):
    def __init__(self, mode):
        # 语义字典
        self.semantics_dict = {
            'living_room': 1,
            'kitchen': 2,
            'bedroom': 3,
            'bathroom': 4,
            'restroom': 5,
            'balcony': 6,
            'closet': 7,
            'corridor': 8,
            'washing_room': 9,
            'PS': 10,
            'outside': 11,
            'wall': 12,
            'no_type': 0
        }
        if mode == 'train':
            self.data = self.load_data('../datasets/lifulldata/annot_json/instances_train.json', '../datasets/lifulldata/annot_npy')
        elif mode == 'val':
            self.data = self.load_data('../datasets/lifulldata/annot_json/instances_val.json', '../datasets/lifulldata/annot_npy')
        elif mode == 'test':
            self.data = self.load_data('../datasets/lifulldata/annot_json/instances_test.json', '../datasets/lifulldata/annot_npy')
        

    def load_data(self, json_path, npy_dir):
        # 读取 JSON 文件
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # 初始化整合数据结构
        integrated_data = {}

        # 整合图像信息
        for image_info in tqdm(data['images']):
            image_id = image_info['id']
            integrated_data[image_id] = {
                'annotations': [],
                'npy_data': None
            }

        # 整合注释信息
        for annotation in tqdm(data['annotations']):
            image_id = annotation['image_id']
            if image_id in integrated_data:
                integrated_data[image_id]['annotations'].append({
                    'point': annotation['point'],
                    'semantic': annotation['semantic']
                })
        
            # 提供的注释数据
            annotations = integrated_data[image_id]['annotations']
            
            # 初始化点集和语义集
            point_set = []
            semantic_set = []
            
            # 遍历注释数据
            for annotation in annotations:
                point = annotation['point']
                semantics = annotation['semantic']
                
                # 添加点到点集
                point_set.append(point)
                
                # 创建multi-hot编码的向量
                semantic_vector = [0] * len(self.semantics_dict)  # 初始化为全0的向量
                for semantic in semantics:
                    if semantic in self.semantics_dict:
                        semantic_index = self.semantics_dict[semantic]
                        semantic_vector[semantic_index] = 1  # 设置对应的索引为1
                
                # 添加语义向量到语义集
                semantic_set.append(semantic_vector)
            assert len(point_set) == len(semantic_set)
            corners_with_semantics = np.concatenate(((np.array(point_set) - 256) / 256, np.array(semantic_set)), axis=1) # 我们只有这里除以256，就当和rplan数据一样了
            corners_with_semantics_pad = np.zeros((53 - len(corners_with_semantics), 15))
            corners_with_semantics = np.concatenate((corners_with_semantics, corners_with_semantics_pad), axis=0)
            integrated_data[image_id]['corners_with_semantics'] = corners_with_semantics

        for image_id in integrated_data.keys():
            integrated_data[image_id].pop('annotations', None)
        # print(self.maxn) 47，我们可以还用之前的53偷个懒
        # 整合 npy 数据
        for image_id in integrated_data.keys():
            npy_path = os.path.join(npy_dir, f"{image_id}.npy")
            if os.path.exists(npy_path):
                npy_data = np.load(npy_path, allow_pickle=True).item()
                # 清理不需要的 'quatree' 键
                npy_data.pop('quatree', None)
                integrated_data[image_id]['npy_data'] = npy_data
                # 获取所有的点并分配索引
                points_npy = list(npy_data.keys())
                point_indices = {point: idx for idx, point in enumerate(points_npy)}
                
                # 初始化邻接矩阵
                n = len(points_npy)
                adjacency_matrix = np.zeros((53, 53), dtype=np.uint8)
                adjacency_matrix[:n, :n] = 1
                edges = np.zeros((53, 53), dtype=np.uint8)

                # 填充邻接矩阵
                for point, neighbors in npy_data.items():
                    for neighbor in neighbors:
                        if neighbor != (-1, -1):  # 忽略不存在的邻居
                            point_idx = point_indices[point]
                            neighbor_idx = point_indices[neighbor]
                            edges[point_idx, neighbor_idx] = 1
                edges = edges.reshape(2809, 1)
                integrated_data[image_id]['adjacency_matrix'] = adjacency_matrix
                integrated_data[image_id]['edges'] = edges
                pdm = np.zeros((53, 1), dtype=np.uint8)
                pdm[:n, :] = 1
                integrated_data[image_id]['padding_mask'] = pdm
        for image_id in integrated_data.keys():
            integrated_data[image_id].pop('npy_data', None)

        
        return integrated_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = list(self.data.keys())[idx]
        d = self.data[image_id]
        corners_withsemantics_simplified = d['corners_with_semantics']
        corners_padding_mask = d['padding_mask']
        edges = d['edges']
        global_attn_matrix = d['adjacency_matrix'].astype(bool)
        
        return corners_withsemantics_simplified, global_attn_matrix, corners_padding_mask, edges