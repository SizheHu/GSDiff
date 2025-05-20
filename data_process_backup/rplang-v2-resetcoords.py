import numpy as np, os
from tqdm import *
for fn in tqdm(os.listdir(r'rplang-v2/train')):
    graph = np.load(os.path.join('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-2/train', fn), allow_pickle=True).item()
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
    edge_coords_padding_mask = graph['global_matrix_np_padding'].reshape(-1, 1)
    corners = np.zeros((53, 2), dtype=graph['corners_np'].dtype)
    corners[0:len(graph['corners_np']), :] = graph['corners_np']
    edge_coords1 = np.repeat(corners[:, None, :], 53, 1).reshape(-1, 2) * edge_coords_padding_mask
    edge_coords2 = np.repeat(corners[None, :, :], 53, 0).reshape(-1, 2) * edge_coords_padding_mask
    print(edge_coords1)
    print(edge_coords2)
    edge_euclidean = np.sqrt(((edge_coords1 - edge_coords2) ** 2)[:, 0:1] + ((edge_coords1 - edge_coords2) ** 2)[:, 1:2])
    print(edge_euclidean)
    assert 0

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

    new_graph['edges'] = graph['edges'].reshape(padding_to_number, padding_to_number, 1)

    np.save('../datasets/rplang-v2-edge/train/' + str(new_graph['file_id']) + '.npy', new_graph)

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

    new_graph['edges'] = graph['edges'].reshape(padding_to_number, padding_to_number, 1)

    np.save('../datasets/rplang-v2-edge/test/' + str(new_graph['file_id']) + '.npy', new_graph)
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