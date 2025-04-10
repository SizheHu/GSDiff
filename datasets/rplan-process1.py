import copy
import os, numpy as np
from tqdm import tqdm
b = np.load('./rplandata/Data/structure_graphs.npy', allow_pickle=True).item()

c = {}
for fn, adjacency_list in tqdm(b.items()):
    graph = {}
    graph['adjacency_list'] = copy.deepcopy(adjacency_list)

    corners_list = list(adjacency_list.keys())
    adjacency_matrix = []
    for i in corners_list:
        adjacency_matrix_i = []
        for j in corners_list:
            if j in adjacency_list[i]:
                adjacency_matrix_i.append(1)
            else:
                adjacency_matrix_i.append(0)
        adjacency_matrix.append(adjacency_matrix_i)
    graph['corners'] = corners_list
    graph['adjacency_matrix'] = adjacency_matrix
    c[fn] = graph

np.save('./rplandata/Data/structure_graphs1.npy', c)
os.rename('./rplandata/Data/structure_graphs1.npy', './rplandata/Data/structure_graphs.npy')

# print(len(b))
# print(b[1])
# print(b[0])
# print(b[999])
# print(b[33333])