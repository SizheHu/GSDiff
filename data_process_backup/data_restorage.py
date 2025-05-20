import numpy as np
from tqdm import *
import os

np.set_printoptions(threshold=np.inf, linewidth=999999)
structure_graphs = np.load('../datasets/structure_graphs.npy', allow_pickle=True).item()


for file_id, structure_graph in tqdm(structure_graphs.items()):
    if file_id % 25 in {0, 1, 2, 3, 4}:
        g = {}
        # print('file_id', file_id)
        g['file_id'] = file_id
        '''original corner data'''
        corners = structure_graph['corners']
        # print('corners', corners)
        g['corners'] = corners
        adjacency_matrix = structure_graph['adjacency_matrix']
        # print('adjacency_matrix', adjacency_matrix)
        g['adjacency_matrix'] = adjacency_matrix
        adjacency_list = structure_graph['adjacency_list']
        # print('adjacency_list', adjacency_list)
        g['adjacency_list'] = adjacency_list
        '''convert to ndarray'''
        corners_np = np.array([list(_) for _ in corners], dtype=np.float64)
        # print('corners_np', corners_np)
        g['corners_np'] = corners_np
        adjacency_matrix_np = np.array(adjacency_matrix, dtype=np.uint8)
        # print('adjacency_matrix_np', adjacency_matrix_np)
        g['adjacency_matrix_np'] = adjacency_matrix_np
        adjacency_list_np = np.array(adjacency_list, dtype=np.uint8)
        # print('adjacency_list_np', adjacency_list_np)
        g['adjacency_list_np'] = adjacency_list_np
        '''normalization (coords: [0, 255] -> [-1, 1])
           if rescale roi(bounding box) to [-1, 1], layouts' large/tiny area will be damaged,
           we want to generate both tiny and large (rational area) layout,
           all rois are already in center(if not in center, learning roi biases on canvas is irrational)'''
        corner_list_np_normalized = (corners_np - 128) / 128
        # print('corner_list_np_normalized', corner_list_np_normalized)
        g['corner_list_np_normalized'] = corner_list_np_normalized
        '''padding and attn mask generating.
           1 means compute and 0 means padding.
           padding to 53 because max corner number is 53; (this is rational!! more paddings make no sense.)
           we don't use 100 because 100*100 edges are too large, about 4 times to 53*53'''
        padding_to_number = 53
        '''after-padding corner lists'''
        corner_list_np_normalized_padding = np.zeros((padding_to_number, 2), dtype=np.float64)
        corner_list_np_normalized_padding[:len(corner_list_np_normalized), :] = corner_list_np_normalized
        # print('corner_list_np_normalized_padding', corner_list_np_normalized_padding)
        g['corner_list_np_normalized_padding'] = corner_list_np_normalized_padding
        ''' padding mask, only real corners (<=> not padding/virtual corners) should compute loss
            we compute it in advance, to save time'''
        padding_mask = np.zeros((padding_to_number, 1), dtype=np.uint8)
        padding_mask[:len(corner_list_np_normalized), :] = 1
        # print('padding_mask', padding_mask)
        g['padding_mask'] = padding_mask
        '''global matrix, (53, 53), each pair of nodes except padding nodes (attention type 1: every)'''
        global_matrix_np_padding = np.zeros((padding_to_number, padding_to_number), dtype=np.uint8)
        global_matrix_np_padding[:len(corner_list_np_normalized), :len(corner_list_np_normalized)] = 1
        # print('global_matrix_np_padding', global_matrix_np_padding)
        g['global_matrix_np_padding'] = global_matrix_np_padding
        '''adjacency matrix, (53, 53), only edge exists == 1 (attention type 2: adjacent)'''
        adjacency_matrix_np_padding = np.zeros((padding_to_number, padding_to_number), dtype=np.uint8)
        adjacency_matrix_np_padding[:len(adjacency_matrix_np), :len(adjacency_matrix_np)] = adjacency_matrix_np
        # print('adjacency_matrix_np_padding', adjacency_matrix_np_padding)
        g['adjacency_matrix_np_padding'] = adjacency_matrix_np_padding
        '''to prepare edge data. we need 53*53 edges, each is [(x1, y1), (x2, y2), 0/1]
        (in accordance with convention in matrix theory, (x1, y1) for row, (x2, y2) for column)
        (if no coords, maybe also rational?)
        if padding corner, (xi, yi) can be any value ((0, 0) for convenience) as they contribute 0 for attn.'''
        edge_coord1 = np.repeat(corner_list_np_normalized_padding[:, None, :], padding_to_number, axis=1)
        edge_coord2 = np.repeat(corner_list_np_normalized_padding[None, :, :], padding_to_number, axis=0)
        edge_coords = np.concatenate((edge_coord1, edge_coord2), axis=2).reshape(-1, 4)
        # print('edge_coords', edge_coords)
        g['edge_coords'] = edge_coords
        '''edges'''
        edges = adjacency_matrix_np_padding[:, :, None].reshape(-1, 1)
        # print('edges', edges)
        g['edges'] = edges
        '''edges global matrix, (2809, 2809), each pair of nodes except padding nodes (attention type 1: every)
        too large, we need to sample this attention adaptively.
        edge adjacency matrix, (2809, 2809), only edges share one same node == 1 (no matter category 0 (hallucinated) or category 1 (real)), else == 0'''
        c_global_mat_flatten = global_matrix_np_padding.reshape(-1)
        c_global_mat_columns = np.repeat(c_global_mat_flatten[:, None], padding_to_number ** 2, axis=1)
        c_global_mat_rows = np.repeat(c_global_mat_flatten[None, :], padding_to_number ** 2, axis=0)
        edges_global_matrix = np.logical_and(c_global_mat_columns, c_global_mat_rows)
        print()
        # print('edges_global_matrix', edges_global_matrix[5])

        # as edges_global_matrix too large, we only storage indices of 1
        g['edges_global_matrix'] = np.argwhere(edges_global_matrix == 1)

        edges_meshgrid = np.stack(
            np.mgrid[0:padding_to_number, 0:padding_to_number, 0:padding_to_number, 0:padding_to_number], axis=-1).reshape(
            (padding_to_number ** 2, padding_to_number ** 2, 4))
        edge_adjacency_mask = ((edges_meshgrid[:, :, 0] != edges_meshgrid[:, :, 1]) &
                               (edges_meshgrid[:, :, 0] == edges_meshgrid[:, :, 2]) &
                               (edges_meshgrid[:, :, 0] != edges_meshgrid[:, :, 3]) &
                               (edges_meshgrid[:, :, 1] != edges_meshgrid[:, :, 2]) &
                               (edges_meshgrid[:, :, 1] != edges_meshgrid[:, :, 3]) &
                               (edges_meshgrid[:, :, 2] != edges_meshgrid[:, :, 3])) | \
                              ((edges_meshgrid[:, :, 0] != edges_meshgrid[:, :, 1]) &
                               (edges_meshgrid[:, :, 0] != edges_meshgrid[:, :, 2]) &
                               (edges_meshgrid[:, :, 0] == edges_meshgrid[:, :, 3]) &
                               (edges_meshgrid[:, :, 1] != edges_meshgrid[:, :, 2]) &
                               (edges_meshgrid[:, :, 1] != edges_meshgrid[:, :, 3]) &
                               (edges_meshgrid[:, :, 2] != edges_meshgrid[:, :, 3])) | \
                              ((edges_meshgrid[:, :, 0] != edges_meshgrid[:, :, 1]) &
                               (edges_meshgrid[:, :, 0] != edges_meshgrid[:, :, 2]) &
                               (edges_meshgrid[:, :, 0] != edges_meshgrid[:, :, 3]) &
                               (edges_meshgrid[:, :, 1] == edges_meshgrid[:, :, 2]) &
                               (edges_meshgrid[:, :, 1] != edges_meshgrid[:, :, 3]) &
                               (edges_meshgrid[:, :, 2] != edges_meshgrid[:, :, 3])) | \
                              ((edges_meshgrid[:, :, 0] != edges_meshgrid[:, :, 1]) &
                               (edges_meshgrid[:, :, 0] != edges_meshgrid[:, :, 2]) &
                               (edges_meshgrid[:, :, 0] != edges_meshgrid[:, :, 3]) &
                               (edges_meshgrid[:, :, 1] != edges_meshgrid[:, :, 2]) &
                               (edges_meshgrid[:, :, 1] == edges_meshgrid[:, :, 3]) &
                               (edges_meshgrid[:, :, 2] != edges_meshgrid[:, :, 3]))
        edges_adjacency_matrix = np.zeros((padding_to_number ** 2, padding_to_number ** 2), dtype=np.uint8)
        edges_adjacency_matrix[np.where(edge_adjacency_mask)] = 1
        edges_adjacency_matrix = np.logical_and(edges_global_matrix, edges_adjacency_matrix)
        # print('edges_adjacency_matrix', edges_adjacency_matrix[5])

        # as edges_adjacency_matrix too large, we only storage indices of 1
        g['edges_adjacency_matrix'] = np.argwhere(edges_adjacency_matrix == 1)
        np.save('../datasets/rplang/' + str(file_id) + '.npy', g)
        del g