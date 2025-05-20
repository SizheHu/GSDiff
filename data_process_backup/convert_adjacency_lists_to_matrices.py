import copy
import os, numpy as np
b = np.load('../datasets/structure_graphs.npy', allow_pickle=True).item()

c = {}
for fn, adjacency_list in b.items():
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

np.save('../datasets/structure_graphs1.npy', c)
d = np.load('../datasets/structure_graphs.npy', allow_pickle=True).item()
print(len(d))
for k, v in d[1].items():
    print(k, v)
#
# os.rename('../datasets/structure_graphs1.npy', '../datasets/structure_graphs.npy')

