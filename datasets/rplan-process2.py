import copy
import os, numpy as np
from tqdm import tqdm
b = np.load('./rplandata/Data/structure_graphs.npy', allow_pickle=True).item()

c = {}
for fn, graph_ori in tqdm(b.items()):
    graph = {}
    graph['corners'] = copy.deepcopy(graph_ori['corners'])
    graph['adjacency_matrix'] = copy.deepcopy(graph_ori['adjacency_matrix'])
    new_adjacency_list = []
    for corner in graph['corners']:
        adj_indices = []
        for direc in range(4):
            if graph_ori['adjacency_list'][corner][direc] == (-1, -1):
                adj_indices.append(-1)
            else:
                adj_indices.append(graph['corners'].index(graph_ori['adjacency_list'][corner][direc]))
        new_adjacency_list.append(adj_indices)
    graph['adjacency_list'] = copy.deepcopy(new_adjacency_list)
    c[fn] = graph

np.save('./rplandata/Data/structure_graphs1.npy', c)
os.rename('./rplandata/Data/structure_graphs1.npy', './rplandata/Data/structure_graphs.npy')

# print(len(b))
# print(b[1])
# print(b[0])
# print(b[999])
# print(b[33333])