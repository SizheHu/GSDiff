import numpy as np, os
from tqdm import *
for fn in tqdm(os.listdir(r'rplang-v2/train')):
    graph = np.load(os.path.join('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2/train', fn), allow_pickle=True).item()
    '''dict_keys([
    'file_id',
    'corners',
    'adjacency_matrix',
    'adjacency_list',
    'corners_np',
    'adjacency_matrix_np',
    'adjacency_list_np',
    'corner_list_np_normalized',
    'corner_list_np_normalized_padding',
    'padding_mask',
    'global_matrix_np_padding',
    'adjacency_matrix_np_padding',
    'edge_coords',
    'edges'])
'''

    new_graph = {}
    new_graph['file_id'] = graph['file_id']
    new_graph['corner_list_np_normalized'] = graph['corners_np'] / 256
    padding_to_number = 53
    corner_list_np_normalized_padding = np.zeros((padding_to_number, 2), dtype=np.float64)
    corner_list_np_normalized_padding[:len(new_graph['corner_list_np_normalized']), :] = new_graph['corner_list_np_normalized']
    new_graph['corner_list_np_normalized_padding'] = corner_list_np_normalized_padding
    del new_graph['corner_list_np_normalized']

    new_graph['padding_mask'] = graph['padding_mask']

    new_graph['global_matrix_np_padding'] = graph['global_matrix_np_padding'].reshape(padding_to_number, padding_to_number, 1).astype(bool)

    np.save('../datasets/rplang-v2-corner/train/' + str(new_graph['file_id']) + '.npy', new_graph)

for fn in tqdm(os.listdir(r'rplang-v2/test')):
    graph = np.load(os.path.join('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2/test', fn), allow_pickle=True).item()
    '''dict_keys([
    'file_id',
    'corners',
    'adjacency_matrix',
    'adjacency_list',
    'corners_np',
    'adjacency_matrix_np',
    'adjacency_list_np',
    'corner_list_np_normalized',
    'corner_list_np_normalized_padding',
    'padding_mask',
    'global_matrix_np_padding',
    'adjacency_matrix_np_padding',
    'edge_coords',
    'edges'])
'''

    new_graph = {}
    new_graph['file_id'] = graph['file_id']
    new_graph['corner_list_np_normalized'] = graph['corners_np'] / 256
    padding_to_number = 53
    corner_list_np_normalized_padding = np.zeros((padding_to_number, 2), dtype=np.float64)
    corner_list_np_normalized_padding[:len(new_graph['corner_list_np_normalized']), :] = new_graph[
        'corner_list_np_normalized']
    new_graph['corner_list_np_normalized_padding'] = corner_list_np_normalized_padding
    del new_graph['corner_list_np_normalized']

    new_graph['padding_mask'] = graph['padding_mask']

    new_graph['global_matrix_np_padding'] = graph['global_matrix_np_padding'].reshape(padding_to_number,
                                                                                      padding_to_number, 1).astype(bool)

    np.save('../datasets/rplang-v2-corner/test/' + str(new_graph['file_id']) + '.npy', new_graph)
import random
print(np.load(os.path.join('/datasets/rplang-v2-edge/test',
                           os.listdir('/datasets/rplang-v2-edge/test')[random.randint(0, 1999)]), allow_pickle=True).item())
print(np.load(os.path.join('/datasets/rplang-v2-edge/train',
                           os.listdir('/datasets/rplang-v2-edge/train')[random.randint(0, 69762)]), allow_pickle=True).item())
print(np.load(os.path.join('/datasets/rplang-v2-edge/test',
                           os.listdir('/datasets/rplang-v2-edge/test')[random.randint(0, 1999)]), allow_pickle=True).item())
print(np.load(os.path.join('/datasets/rplang-v2-edge/train',
                           os.listdir('/datasets/rplang-v2-edge/train')[random.randint(0, 69762)]), allow_pickle=True).item())
print(np.load(os.path.join('/datasets/rplang-v2-edge/test',
                           os.listdir('/datasets/rplang-v2-edge/test')[random.randint(0, 1999)]), allow_pickle=True).item())
print(np.load(os.path.join('/datasets/rplang-v2-edge/train',
                           os.listdir('/datasets/rplang-v2-edge/train')[random.randint(0, 69762)]), allow_pickle=True).item())
print(np.load(os.path.join('/datasets/rplang-v2-edge/test',
                           os.listdir('/datasets/rplang-v2-edge/test')[random.randint(0, 1999)]), allow_pickle=True).item())
print(np.load(os.path.join('/datasets/rplang-v2-edge/train',
                           os.listdir('/datasets/rplang-v2-edge/train')[random.randint(0, 69762)]), allow_pickle=True).item())
print(np.load(os.path.join('/datasets/rplang-v2-edge/test',
                           os.listdir('/datasets/rplang-v2-edge/test')[random.randint(0, 1999)]), allow_pickle=True).item())
print(np.load(os.path.join('/datasets/rplang-v2-edge/train',
                           os.listdir('/datasets/rplang-v2-edge/train')[random.randint(0, 69762)]), allow_pickle=True).item())