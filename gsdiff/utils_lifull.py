import copy
import math
import os
import random
import networkx as nx
import torch
from tqdm import *
import cv2
import numpy as np


def euclidean_edge_match(input_array):
    # 目标值数组
    target_values = np.array([[0, 1], [1, 0]])
    # 计算欧氏距离
    distances = np.linalg.norm(input_array[:, None, :] - target_values, axis=-1)
    # 找到距离最小值的索引
    min_indices = np.argmin(distances, axis=-1)
    # 使用索引映射到目标值
    output_array = target_values[min_indices]
    return output_array

def visualize(result_corners_inverse_normalized, result_edges_unpaddinged, output_dir, timestep):
    '''timestep can be 'gt' or any int time'''
    for result_index, result in enumerate(result_corners_inverse_normalized):
        result = result[0]
        # print(result)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        # drew walls
        drew_walls = [] 
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            # draw walls
            if result_edges_unpaddinged[result_index][0].shape[1] == 1:
                edges = result_edges_unpaddinged[result_index][0][:, 0]
            else:
                edges = result_edges_unpaddinged[result_index][0][:, 1]
            for edge_index, one_or_zero in enumerate(edges):
                if one_or_zero:
                    if corner_index in [edge_index % len(result), edge_index // len(result)] and (edge_index % len(result) != edge_index // len(result)):
                        # get another corner
                        another_corner = result[(edge_index % len(result)) if corner_index == (edge_index // len(result)) else (edge_index // len(result))]
                        another_corner_tuple = tuple(another_corner.tolist())
                        if (corner_tuple, another_corner_tuple) in drew_walls or (
                        another_corner_tuple, corner_tuple) in drew_walls:
                            pass
                        else:
                            # print(corner_tuple, another_corner_tuple)
                            cv2.line(img, corner_tuple, another_corner_tuple, color=(0, 0, 0), thickness=3)
                            drew_walls.append((corner_tuple, another_corner_tuple))
                            drew_walls.append((another_corner_tuple, corner_tuple))
        # draw corners
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
        cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)

def visualize_withsemantic(output_points, output_edges, simple_cycles, simple_cycles_semantics, output_dir, result_index, timestep):
    '''timestep can be 'gt' or any int time'''
    colors = {6: (0, 0, 0), 0: (222, 241, 244), 1: (159, 182, 234), 2: (92, 112, 107), 3: (95, 122, 224),
              4: (123, 121, 95), 5: (143, 204, 242)}
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    # draw polygons
    for polygon_i, polygon in enumerate(simple_cycles):
        # 提取有效的x, y坐标
        pts = np.array([(x, y) for x, y, *rest in polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # 画多边形
        cv2.fillPoly(img, [pts], color=colors[simple_cycles_semantics[polygon_i]])
        cv2.polylines(img, [pts], isClosed=True, color=(150, 150, 150), thickness=5)
    # draw corners
    # for corner in output_points:
    #     corner_tuple = tuple(list(corner)[:2])
    #     cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
    cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)

def visualize_35(result_corners_inverse_normalized, result_edges_unpaddinged, output_dir, timestep):
    '''timestep can be 'gt' or any int time'''
    for result_index, result in enumerate(result_corners_inverse_normalized):
        result = result[0]
        # print(result)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        # drew walls
        drew_walls = []
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            # draw walls
            if result_edges_unpaddinged[result_index][0].shape[1] == 1:
                edges = result_edges_unpaddinged[result_index][0][:, 0]
            else:
                edges = result_edges_unpaddinged[result_index][0][:, 1]
            for edge_index, one_or_zero in enumerate(edges):
                if one_or_zero:
                    if corner_index in [edge_index % len(result), edge_index // len(result)] and (edge_index % len(result) != edge_index // len(result)):
                        # get another corner
                        another_corner = result[(edge_index % len(result)) if corner_index == (edge_index // len(result)) else (edge_index // len(result))]
                        another_corner_tuple = tuple(another_corner.tolist())
                        if (corner_tuple, another_corner_tuple) in drew_walls or (
                        another_corner_tuple, corner_tuple) in drew_walls:
                            pass
                        else:
                            # print(corner_tuple, another_corner_tuple)
                            cv2.line(img, corner_tuple, another_corner_tuple, color=(0, 0, 0), thickness=3)
                            drew_walls.append((corner_tuple, another_corner_tuple))
                            drew_walls.append((another_corner_tuple, corner_tuple))
        # draw corners
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
        cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)

def visualize_36(result_corners_inverse_normalized, result_edges_unpaddinged, output_dir, timestep):
    '''timestep can be 'gt' or any int time'''
    for result_index, result in enumerate(result_corners_inverse_normalized):
        result = result[0]
        # print(result)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        # draw corners
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
        cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)

def visualize_51(result_corners_inverse_normalized, result_semantics_inverse_normalized, output_dir, timestep):
    '''timestep can be 'gt' or any int time'''
    colors = {13:(0, 0, 0), 0:(255, 0, 0), 1:(0, 255, 0), 2:(0, 0, 255), 3:(255, 255, 0),
              4:(255, 0, 255), 5:(0, 255, 255), 6:(127, 0, 0), 7:(0, 127, 0),
              8:(0, 0, 127), 9:(127, 127, 0), 10:(127, 0, 127), 11:(0, 127, 127), 12:(127, 127, 127)}

    for result_index, result in enumerate(result_corners_inverse_normalized):
        result = result[0]
        # print(result)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        # draw corners
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
            # draw semantics
            if timestep == 'gt' or timestep == 0:
                semantic = result_semantics_inverse_normalized[result_index][0][corner_index].tolist()
                # print(semantic)
                semans = []
                for seman_i in range(14):
                    if semantic[seman_i] > 0:
                        for _ in range(semantic[seman_i]):
                            semans.append(colors[seman_i])
                # print(semans)
                # print(corner_tuple)
                for ii, seman in enumerate(semans):
                    img[corner_tuple[1] + 4, corner_tuple[0] + ii, 0] = seman[0]
                    img[corner_tuple[1] + 4, corner_tuple[0] + ii, 1] = seman[1]
                    img[corner_tuple[1] + 4, corner_tuple[0] + ii, 2] = seman[2]

        cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)

def visualize_33(result_corners_inverse_normalized, result_edges_unpaddinged, output_dir, timestep):
    # print(len(result_edges_unpaddinged)) # bs
    # print(len(result_edges_unpaddinged[0]), len(result_edges_unpaddinged[1])) # 1, 1
    # print(len(result_edges_unpaddinged[0][0]), len(result_edges_unpaddinged[1][0])) # 18 ** 2, 22 ** 2
    # print(result_edges_unpaddinged[0][0][-1], result_edges_unpaddinged[1][0][-1])  # 18 ** 2, 22 ** 2
    # 需要映射到[0 1]和[1 0]欧氏距离更近的那个点
    result_edges_unpaddinged_discrete = []
    for bs_i in range(len(result_edges_unpaddinged)):
        aaa = []
        for _ in range(1):
            e = euclidean_edge_match(result_edges_unpaddinged[bs_i][_])
            aaa.append(e)
        result_edges_unpaddinged_discrete.append(aaa)
    result_edges_unpaddinged = result_edges_unpaddinged_discrete

    '''timestep can be 'gt' or any int time'''
    for result_index, result in enumerate(result_corners_inverse_normalized):
        result = result[0]
        # print(result)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        # drew walls
        drew_walls = []
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            # draw walls
            if result_edges_unpaddinged[result_index][0].shape[1] == 1:
                edges = result_edges_unpaddinged[result_index][0][:, 0]
            else:
                edges = result_edges_unpaddinged[result_index][0][:, 1]
            for edge_index, one_or_zero in enumerate(edges):
                if one_or_zero:
                    if corner_index in [edge_index % len(result), edge_index // len(result)] and (edge_index % len(result) != edge_index // len(result)):
                        # get another corner
                        another_corner = result[(edge_index % len(result)) if corner_index == (edge_index // len(result)) else (edge_index // len(result))]
                        another_corner_tuple = tuple(another_corner.tolist())
                        if (corner_tuple, another_corner_tuple) in drew_walls or (
                        another_corner_tuple, corner_tuple) in drew_walls:
                            pass
                        else:
                            # print(corner_tuple, another_corner_tuple)
                            cv2.line(img, corner_tuple, another_corner_tuple, color=(0, 0, 0), thickness=3)
                            drew_walls.append((corner_tuple, another_corner_tuple))
                            drew_walls.append((another_corner_tuple, corner_tuple))
        # draw corners
        for corner_index, corner in enumerate(result):
            corner_tuple = tuple(corner.tolist())
            cv2.circle(img, corner_tuple, radius=3, color=(0, 215, 255), thickness=-1)
        cv2.imwrite(output_dir + str(result_index) + '_' + str(timestep) + '.png', img)


def chemistry_visualize(result_atoms_unpaddinged, result_bonds_unpaddinged, output_dir, timestep):
    print(timestep)
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    sample_numbers = len(result_atoms_unpaddinged)
    for i in range(sample_numbers):
        print(i)
        print(result_atoms_unpaddinged[i])
        print(result_bonds_unpaddinged[i].reshape(1, len(result_atoms_unpaddinged[i][0]), len(result_atoms_unpaddinged[i][0]), 5))

        # 原子类型和键类型的 one-hot ndarray
        atom_array = result_atoms_unpaddinged[i][0]  # Replace with your atom array
        bond_array = result_bonds_unpaddinged[i][0]  # Replace with your bond array

        # 定义原子符号列表
        atom_symbols = ["C", "N", "O", "F"]

        # 将 one-hot ndarray 转换为分子结构
        def ndarray_to_molecule(atom_array, bond_array):
            # 创建原子列表
            atoms = [atom_symbols[np.argmax(atom_one_hot)] for atom_one_hot in atom_array]

            # 创建空的可编辑分子
            mol = Chem.EditableMol(Chem.Mol())

            # 添加原子
            for atom in atoms:
                mol.AddAtom(Chem.Atom(atom))

            # 添加键(upper trim)
            num_atoms = len(atom_array)
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    bond_type = np.argmax(bond_array[i * num_atoms + j])  # 获取键类型（0：无键，1：单键，2：双键，3：三键）
                    if bond_type > 0:
                        mol.AddBond(i, j, Chem.BondType.values[bond_type])


            # 获取不可编辑的分子
            mol = mol.GetMol()
            for a in mol.GetAtoms():
                print(a.GetIdx(), a.GetSymbol())
            for b in mol.GetBonds():
                print(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())

            return mol

        mol = ndarray_to_molecule(atom_array, bond_array)
        from rdkit.Chem import Draw

        # 生成2D坐标
        AllChem.Compute2DCoords(mol)

        # 展示分子
        Draw.MolToFile(mol, output_dir + str(i) + '_' + str(timestep) + '.png', size=(256, 256))

def inverse_normalize_remove_padding(result_corners, results_edges, results_corners_numbers):
    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * 128 + 128).cpu()).astype(np.uint8)[:,
                                    :results_corners_numbers[i], :]
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_edges_unpaddinged = []
    for j, l in enumerate(results_edges):
        result_edge_unpaddinged = []
        for _ in range(53):
            if _ < results_corners_numbers[j]:
                result_edge_unpaddinged.append(np.array(l.cpu())[:, _ * 53:_ * 53 + results_corners_numbers[j], :])
        result_edge_unpaddinged = np.concatenate(result_edge_unpaddinged, axis=1)
        result_edges_unpaddinged.append(result_edge_unpaddinged)
    return result_corners_inverse_normalized, result_edges_unpaddinged

def edges_remove_padding(results_edges, results_corners_numbers):
    result_edges_unpaddinged = []
    for j, l in enumerate(results_edges):
        result_edge_unpaddinged = []
        for _ in range(53):
            if _ < results_corners_numbers[j]:
                result_edge_unpaddinged.append(np.array(l.cpu())[:, _ * 53:_ * 53 + results_corners_numbers[j], :])
        result_edge_unpaddinged = np.concatenate(result_edge_unpaddinged, axis=1)
        result_edges_unpaddinged.append(result_edge_unpaddinged)
    return result_edges_unpaddinged

def inverse_normalize_remove_padding_51(result_corners, result_semantics, results_corners_numbers):
    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * 128 + 128).cpu()).astype(np.uint8)[:,
                                    :results_corners_numbers[i], :]
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_semantics_inverse_normalized = []
    for i, result in enumerate(result_semantics):
        result_semantic_inverse_normalized = np.round(np.array(result.cpu())).astype(np.int8)[:,
                                           :results_corners_numbers[i], :]
        result_semantics_inverse_normalized.append(result_semantic_inverse_normalized)
    return result_corners_inverse_normalized, result_semantics_inverse_normalized

def inverse_normalize_and_remove_padding(result_corners, result_semantics, results_corners_numbers):
    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * 128 + 128).cpu()).astype(np.uint8)[:,
                                    :results_corners_numbers[i], :]
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_semantics_inverse_normalized = []
    for i, result in enumerate(result_semantics):
        result_semantic_inverse_normalized = np.round(np.array(result.cpu())).astype(np.int8)[:,
                                           :results_corners_numbers[i], :]
        result_semantics_inverse_normalized.append(result_semantic_inverse_normalized)
    return result_corners_inverse_normalized, result_semantics_inverse_normalized

def inverse_normalize_and_remove_padding_100(result_corners, result_semantics, results_corners_numbers):
    # print(result_corners) # 长度3000, 每个元素(1, 53, 2) tensor
    # print(result_semantics) # 长度3000, 每个元素(1, 53, 8) tensor
    # print(results_corners_numbers) # 长度3000, 每个元素(53, ) tensor, 里面的值是0代表是角点，1代表非角点


    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * 128 + 128).cpu()).astype(np.uint8)[:, np.where((results_corners_numbers[i]).cpu().numpy() == 0)[0], :]
        # print(result_corner_inverse_normalized)
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_semantics_inverse_normalized = []
    for i, result in enumerate(result_semantics):
        result_semantic_inverse_normalized = np.round(np.array(result.cpu())).astype(np.int8)[:, np.where((results_corners_numbers[i]).cpu().numpy() == 0)[0], :]
        result_semantics_inverse_normalized.append(result_semantic_inverse_normalized)
    return result_corners_inverse_normalized, result_semantics_inverse_normalized

def inverse_normalize_and_remove_padding_4testing(result_corners, result_semantics, results_corners_numbers, resolution=512):
    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * (resolution // 2) + (resolution // 2)).cpu()).astype(np.int32)[:,
                                    :results_corners_numbers[i], :]
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_semantics_inverse_normalized = []
    for i, result in enumerate(result_semantics):
        result_semantic_inverse_normalized = np.round(np.array(result.cpu())).astype(np.int8)[:,
                                           :results_corners_numbers[i], :]
        result_semantics_inverse_normalized.append(result_semantic_inverse_normalized)
    return result_corners_inverse_normalized, result_semantics_inverse_normalized

def inverse_normalize_and_remove_padding_100_4testing(result_corners, result_semantics, results_corners_numbers, resolution=512):
    # print(result_corners) # 长度3000, 每个元素(1, 53, 2) tensor
    # print(result_semantics) # 长度3000, 每个元素(1, 53, 8) tensor
    # print(results_corners_numbers) # 长度3000, 每个元素(53, ) tensor, 里面的值是0代表是角点，1代表非角点

    result_corners_inverse_normalized = []
    for i, result in enumerate(result_corners):
        result_corner_inverse_normalized = np.array((result * (resolution // 2) + (resolution // 2)).cpu()).astype(np.int32)[:, np.where((results_corners_numbers[i]).cpu().numpy() == 0)[0], :]
        # print(result_corner_inverse_normalized)
        result_corners_inverse_normalized.append(result_corner_inverse_normalized)

    result_semantics_inverse_normalized = []
    for i, result in enumerate(result_semantics):
        result_semantic_inverse_normalized = np.round(np.array(result.cpu())).astype(np.int8)[:, np.where((results_corners_numbers[i]).cpu().numpy() == 0)[0], :]
        result_semantics_inverse_normalized.append(result_semantic_inverse_normalized)
    return result_corners_inverse_normalized, result_semantics_inverse_normalized


def chemistry_remove_padding(result_atoms, result_bonds, results_atoms_numbers):
    result_atoms_unpaddinged = []
    for i, k in enumerate(result_atoms):
        result_atoms_unpaddinged.append(np.array(k.cpu()).astype(np.uint8)[:,
                                    :results_atoms_numbers[i], :])

    result_bonds_unpaddinged = []
    for j, l in enumerate(result_bonds):
        result_bond_unpaddinged = []
        for _ in range(9):
            if _ < results_atoms_numbers[j]:
                result_bond_unpaddinged.append(np.array(l.cpu())[:, _ * 9:_ * 9 + results_atoms_numbers[j], :])
        result_bond_unpaddinged = np.concatenate(result_bond_unpaddinged, axis=1)
        result_bonds_unpaddinged.append(result_bond_unpaddinged)

    return result_atoms_unpaddinged, result_bonds_unpaddinged

def edges_prior_distribution():
    # edges_category1_count = 0
    # edges_category2_count = 0
    # for f in tqdm(os.listdir('../datasets/rplang/train')):
    #     g = np.load('../datasets/rplang/train/' + f, allow_pickle=True).item()
    #     e1_and_2 = g['global_matrix_np_padding']
    #     e2 = g['adjacency_matrix_np_padding']
    #     e2_number = np.sum(e2)
    #     e1_and_2_number = np.sum(e1_and_2)
    #     e1_number = e1_and_2_number - e2_number
    #     edges_category1_count += e1_number
    #     edges_category2_count += e2_number
    # for f in tqdm(os.listdir('../datasets/rplang/test')):
    #     g = np.load('../datasets/rplang/test/' + f, allow_pickle=True).item()
    #     e1_and_2 = g['global_matrix_np_padding']
    #     e2 = g['adjacency_matrix_np_padding']
    #     e2_number = np.sum(e2)
    #     e1_and_2_number = np.sum(e1_and_2)
    #     e1_number = e1_and_2_number - e2_number
    #     edges_category1_count += e1_number
    #     edges_category2_count += e2_number
    # print(edges_category1_count, edges_category2_count)
    # return [edges_category1_count / (edges_category1_count + edges_category2_count),
    #         edges_category2_count / (edges_category1_count + edges_category2_count)]
    return np.array([0.8894, 0.1106], dtype=np.float64)

def atoms_prior_distribution():
    return np.array([0.7230, 0.1151, 0.1593, 0.0026], dtype=np.float64)

def bonds_prior_distribution():
    return np.array([0.7261, 0.2384, 0.0274, 0.0081, 0.0], dtype=np.float64)

def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == False] = -float("inf")
    return torch.softmax(x_masked, **kwargs)

def get_quadrant(angle):
    # 角占每个象限的多少度（理论上两个向量不会重合）
    if angle[0] < angle[1]: # 角1
        if 0 <= angle[0] < 90 and 0 <= angle[1] < 90:
            quadrant = (angle[1] - angle[0], 0, 0, 0)
        elif 0 <= angle[0] < 90 and 90 <= angle[1] < 180:
            quadrant = (90 - angle[0], angle[1] - 90, 0, 0)
        elif 0 <= angle[0] < 90 and 180 <= angle[1] < 270:
            quadrant = (90 - angle[0], 90, angle[1] - 180, 0)
        elif 0 <= angle[0] < 90 and 270 <= angle[1] < 360:
            quadrant = (90 - angle[0], 90, 90, angle[1] - 270)
        elif 90 <= angle[0] < 180 and 90 <= angle[1] < 180:
            quadrant = (0, angle[1] - angle[0], 0, 0)
        elif 90 <= angle[0] < 180 and 180 <= angle[1] < 270:
            quadrant = (0, 180 - angle[0], angle[1] - 180, 0)
        elif 90 <= angle[0] < 180 and 270 <= angle[1] < 360:
            quadrant = (0, 180 - angle[0], 90, angle[1] - 270)
        elif 180 <= angle[0] < 270 and 180 <= angle[1] < 270:
            quadrant = (0, 0, angle[1] - angle[0], 0)
        elif 180 <= angle[0] < 270 and 270 <= angle[1] < 360:
            quadrant = (0, 0, 270 - angle[0], angle[1] - 270)
        elif 270 <= angle[0] < 360 and 270 <= angle[1] < 360:
            quadrant = (0, 0, 0, angle[1] - angle[0])
    else: # 角2
        if 0 <= angle[1] < 90 and 0 <= angle[0] < 90:
            quadrant_ = (angle[0] - angle[1], 0, 0, 0)
        elif 0 <= angle[1] < 90 and 90 <= angle[0] < 180:
            quadrant_ = (90 - angle[1], angle[0] - 90, 0, 0)
        elif 0 <= angle[1] < 90 and 180 <= angle[0] < 270:
            quadrant_ = (90 - angle[1], 90, angle[0] - 180, 0)
        elif 0 <= angle[1] < 90 and 270 <= angle[0] < 360:
            quadrant_ = (90 - angle[1], 90, 90, angle[0] - 270)
        elif 90 <= angle[1] < 180 and 90 <= angle[0] < 180:
            quadrant_ = (0, angle[0] - angle[1], 0, 0)
        elif 90 <= angle[1] < 180 and 180 <= angle[0] < 270:
            quadrant_ = (0, 180 - angle[1], angle[0] - 180, 0)
        elif 90 <= angle[1] < 180 and 270 <= angle[0] < 360:
            quadrant_ = (0, 180 - angle[1], 90, angle[0] - 270)
        elif 180 <= angle[1] < 270 and 180 <= angle[0] < 270:
            quadrant_ = (0, 0, angle[0] - angle[1], 0)
        elif 180 <= angle[1] < 270 and 270 <= angle[0] < 360:
            quadrant_ = (0, 0, 270 - angle[1], angle[0] - 270)
        elif 270 <= angle[1] < 360 and 270 <= angle[0] < 360:
            quadrant_ = (0, 0, 0, angle[0] - angle[1])
        quadrant = (90 - quadrant_[0], 90 - quadrant_[1], 90 - quadrant_[2], 90 - quadrant_[3])
    return quadrant

def poly_area(points): # 计算逆时针凹多边形面积，顺时针则为负数
    s = 0
    points_count = len(points)
    for i in range(points_count):
        point = points[i]
        point2 = points[(i + 1) % points_count]
        s += (point[0] - point2[0]) * (point[1] + point2[1])
    return s / 2

def rotate_degree_clockwise_from_counter_degree(src_degree, dest_degree):
    delta = src_degree - dest_degree
    return delta if delta >= 0 else 360 + delta

def rotate_degree_counterclockwise_from_counter_degree(src_degree, dest_degree):
    delta = dest_degree - src_degree
    return delta if delta >= 0 else 360 + delta


def x_axis_angle(y):
    # 以图像坐标系为准，(1,0)方向记为0度，逆时针绕一圈到360度
    # print('-------------')
    # print(y)
    y_right_hand = (y[0], -y[1])
    # print(y_right_hand)

    x = (1, 0)
    inner = x[0] * y_right_hand[0] + x[1] * y_right_hand[1]
    # print(inner)
    y_norm2 = (y_right_hand[0] ** 2 + y_right_hand[1] ** 2) ** 0.5
    # print(y_norm2)
    cosxy = inner / y_norm2
    # print(cosxy)
    angle = math.acos(cosxy)
    # print(angle, math.degrees(angle))
    # print('-------------')
    return math.degrees(angle) if y_right_hand[1] >= 0 else 360 - math.degrees(angle)

def get_results_float_with_semantic(best_result):
    if 1:
        preds = best_result[2]
        # 所有点、边
        output_points = []
        output_edges = []
        for triplet in preds:
            this_preds = triplet[0]
            last_edges = triplet[1]
            this_edges = triplet[2]
            for this_pred in this_preds:
                point = (this_pred['points'].tolist()[0], this_pred['points'].tolist()[1],
                         this_pred['semantic_left_up'].item(), this_pred['semantic_right_up'].item(),
                         this_pred['semantic_right_down'].item(), this_pred['semantic_left_down'].item())
                output_points.append(point)
            for last_edge in last_edges:
                point1 = (last_edge[0]['points'].tolist()[0], last_edge[0]['points'].tolist()[1],
                         last_edge[0]['semantic_left_up'].item(), last_edge[0]['semantic_right_up'].item(),
                         last_edge[0]['semantic_right_down'].item(), last_edge[0]['semantic_left_down'].item())
                point2 = (last_edge[1]['points'].tolist()[0], last_edge[1]['points'].tolist()[1],
                          last_edge[1]['semantic_left_up'].item(), last_edge[1]['semantic_right_up'].item(),
                          last_edge[1]['semantic_right_down'].item(), last_edge[1]['semantic_left_down'].item())
                edge = (point1, point2)
                output_edges.append(edge)
            for this_edge in this_edges:
                point1 = (this_edge[0]['points'].tolist()[0], this_edge[0]['points'].tolist()[1],
                          this_edge[0]['semantic_left_up'].item(), this_edge[0]['semantic_right_up'].item(),
                          this_edge[0]['semantic_right_down'].item(), this_edge[0]['semantic_left_down'].item())
                point2 = (this_edge[1]['points'].tolist()[0], this_edge[1]['points'].tolist()[1],
                          this_edge[1]['semantic_left_up'].item(), this_edge[1]['semantic_right_up'].item(),
                          this_edge[1]['semantic_right_down'].item(), this_edge[1]['semantic_left_down'].item())
                edge = (point1, point2)
                output_edges.append(edge)
        return output_points, output_edges

def get_cycle_basis_and_semantic_2(output_points, output_edges):
    # 一个关于索引和输出点的字典
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # 这里无法处理重复点，不能去掉nms
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # 这里无法处理重复点，不能去掉nms
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))


    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])
        G.add_edge(e[1], e[0])



    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []
    # print('断点1', simple_cycles)
    bridges = list(nx.bridges(G))
    # 虚边怎么做呢：还是直接删掉，从边集中删掉，然后作为多个连通的预测处理即可
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            G.remove_edge(b[1], b[0])
    # 断掉割边集以后，我们查看剩下的所有连通分量，此时只存在孤立点或圈，遍历剩下的所有圈
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            # 获取对应的点边集
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # 固定的边集，不会删除
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # 用于遍历以减少时间复杂度的边集，其中的边会删除
            # print(output_points_c)
            # print(output_edges_c)

            # 求该连通分量所有逆时针simple cycles的方法
            # 定义d中的编号为该连通分量中点的编号，
            # 遍历无向边集output_edges_c：
            # 对每条无向边，规定初始点为编号较小者，上一个点为初始点，当前点为编号较大者；
            # 求当前点的所有邻边，及对应的出射角度
            # 求上一个点入射到当前点的方向的反方向，对应的角度（最常见的极坐标系[0,2pi)）
            # 求从这个角度开始逆时针旋转，碰到的最后一个当前点的邻边
            # 将该邻边的另一端作为下一个点
            # 当下一个点等于初始点时，得到形式类似[p0,p1,...,pn-1,p0]的cycle
            # 检索cycle，从剩下的边中删除所有pi<pi+1的边（包括本边），遍历剩下的边
            # 将上一个点设为当前点，当前点设为下一个点

            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
                    simple_cycle_semantics = []
                    simple_cycle = []
                    simple_cycle_number = []
                    point1 = edge_c[0]
                    point2 = edge_c[1]
                    point1_number = d[point1]
                    point2_number = d[point2]
                    # 初始点
                    initial_point = None
                    initial_point_number = None
                    if point1_number < point2_number:
                        initial_point = point1
                        initial_point_number = point1_number
                    else:
                        initial_point = point2
                        initial_point_number = point2_number
                    simple_cycle.append(initial_point)
                    simple_cycle_number.append(initial_point_number)
                    # 上一个点
                    last_point = initial_point
                    last_point_number = initial_point_number
                    # 当前点
                    current_point = None
                    current_point_number = None
                    if point1_number < point2_number:
                        current_point = point2
                        current_point_number = point2_number
                    else:
                        current_point = point1
                        current_point_number = point1_number
                    simple_cycle.append(current_point)
                    simple_cycle_number.append(current_point_number)
                    # 初始点的后一个点（用于判断while结束）
                    next_initial_point = copy.deepcopy(current_point)
                    next_initial_point_number = copy.deepcopy(current_point_number)
                    # 下一个点
                    next_point = None
                    next_point_number = None
                    # 当下一个点等于初始点的后一个点时，结束
                    while next_point != next_initial_point:
                        # 求当前点的所有邻边
                        relevant_edges = []
                        for edge in output_edges_c:
                            if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                relevant_edges.append(edge)
                        # 求当前点的所有邻边对应的出射角度
                        relevant_edges_degree = []
                        for relevant_edge in relevant_edges:
                            # 出射向量
                            vec = None
                            if relevant_edge[0] == current_point:
                                vec = (
                                relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                            elif relevant_edge[1] == current_point:
                                vec = (
                                relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                            else:
                                assert 0
                            # 求出射角度
                            vec_degree = x_axis_angle(vec)
                            relevant_edges_degree.append(vec_degree)
                        # 求上一个点入射到当前点的方向的反方向（出射方向）、对应出射角度
                        vec_from_current_point_to_last_point = None
                        vec_from_current_point_to_last_point_degree = None
                        for relevant_edge_ind, relevant_edge in enumerate(relevant_edges):
                            if relevant_edge == (current_point, last_point):
                                vec_from_current_point_to_last_point = (
                                relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                relevant_edges.remove(relevant_edge)
                                relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                            elif relevant_edge == (last_point, current_point):
                                vec_from_current_point_to_last_point = (
                                relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                relevant_edges.remove(relevant_edge)
                                relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                            else:
                                continue
                        # 求从这个角度开始逆时针旋转，碰到的最后一个当前点的邻边
                        # 同时把没扫过部分（就是内角）的角度区域记录下来
                        # 这里我们的内角语义替换为完全语义
                        rotate_deltas_counterclockwise = []
                        # 记录内角区域，逆时针，从前一个角度到后一个角度
                        interior_angles = []
                        for relevant_edge_degree in relevant_edges_degree:
                            rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                            rotate_deltas_counterclockwise.append(rotate_delta)
                            interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                        # print(rotate_deltas_counterclockwise)
                        # 最大角对应索引
                        max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                        # 找到对应的内角
                        interior_angle_counterclockwise = interior_angles[max_rotate_index]
                        # 求出对应的语义区域
                        # 先求出当前点的所有语义，顺序按照四个象限排序
                        # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                        #                           current_point[4], ]
                        current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                  current_point[4], current_point[6], current_point[7],
                                                  current_point[8], current_point[9], current_point[10],
                                                  current_point[11], current_point[12], current_point[13],
                                                  current_point[14], current_point[15]]
                        # 求出该逆时针角占了四个象限的多少角度
                        # 求法：直接求度数较小的逆时针转到度数较大的覆盖四象限角度
                        # 然后判断，如果度数较小的是内角区域的“源方向”，则正好是覆盖四象限角度；
                        # 如果度数较小的是内角区域的“目标方向”，则对覆盖四象限角度用90度减去；
                        interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)  # 度数较小的
                        interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # 度数较大的
                        quadrant_smaller_to_bigger_counterclockwise = get_quadrant(
                            (interior_angle_counterclockwise_degree_smaller,
                             interior_angle_counterclockwise_degree_bigger))
                        # print(quadrant_smaller_to_bigger_counterclockwise)
                        if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                            pass
                        elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                            quadrant_smaller_to_bigger_counterclockwise = (
                            90 - quadrant_smaller_to_bigger_counterclockwise[0],
                            90 - quadrant_smaller_to_bigger_counterclockwise[1],
                            90 - quadrant_smaller_to_bigger_counterclockwise[2],
                            90 - quadrant_smaller_to_bigger_counterclockwise[3])
                        else:
                            assert 0
                        # 永远不将语义置为-1
                        current_point_semantic_valid = []
                        for qd, seman in enumerate(current_point_semantic):
                            if 1:
                                current_point_semantic_valid.append(seman)
                            else:
                                current_point_semantic_valid.append(-1)
                        # 对全部语义进行统计
                        simple_cycle_semantics.append(current_point_semantic_valid)

                        # 对应边
                        max_rotate_edge = relevant_edges[max_rotate_index]
                        # 对应下一个点
                        if max_rotate_edge[0] == current_point:
                            next_point = max_rotate_edge[1]
                            next_point_number = d[next_point]
                        elif max_rotate_edge[1] == current_point:
                            next_point = max_rotate_edge[0]
                            next_point_number = d[next_point]
                        else:
                            assert 0
                        # 重新给上一个点、当前点、下一个点赋值，并将当前点加入simple_cycle
                        last_point = current_point
                        last_point_number = current_point_number
                        current_point = next_point
                        current_point_number = next_point_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)
                    # 最后加上初始点（为了删边）
                    # simple_cycle.append(initial_point)
                    # simple_cycle_number.append(initial_point_number)
                    # 检索simple_cycle_number，从剩下的边中删除所有pi<pi+1的边（包括本边）
                    # print('------------------')
                    # print(simple_cycle)
                    # print(simple_cycle_number)
                    # print('------------------')
                    for point_number_ind, point_number in enumerate(simple_cycle_number):
                        if point_number_ind < len(simple_cycle_number) - 1:
                            edge_number = (point_number, simple_cycle_number[point_number_ind + 1])
                            # print(simple_cycle_number)
                            if edge_number[0] < edge_number[1]:
                                if (d_rev[edge_number[0]], d_rev[edge_number[1]]) in output_edges_c_copy_for_traversing:
                                    output_edges_c_copy_for_traversing.remove(
                                        (d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                elif (
                                d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                    output_edges_c_copy_for_traversing.remove(
                                        (d_rev[edge_number[1]], d_rev[edge_number[0]]))
                    # 算面积时不需要闭环
                    simple_cycle.pop(-1)
                    simple_cycle_number.pop(-1)
                    # 存起来（逆时针计算面积，如果面积为负则不加入，说明是最大的那个）
                    polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                    polygon_counterclockwise.pop(-1)
                    # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                    if poly_area(polygon_counterclockwise) > 0:
                        simple_cycles_c.append(simple_cycle)
                        simple_cycles_number_c.append(simple_cycle_number)
                        # 公共最大语义（最大的那个圈就不用算了），得到该simple_cycle的语义并记录
                        semantic_result = {}
                        for semantic_label in range(0, 14):
                            semantic_result[semantic_label] = 0
                        for everypoint_semantic in simple_cycle_semantics:
                            for _ in range(0, 14):
                                if _ in everypoint_semantic:
                                    semantic_result[_] += 1
                        del semantic_result[13]

                        # print(semantic_result)
                        # 如果最高票相同则等概率随机选一个（注意13不算）
                        this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                        # print(this_cycle_semantic)
                        this_cycle_result = None
                        if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                            # 以唯一最高票为准
                            this_cycle_result = this_cycle_semantic[0][0]
                        else:
                            # 找出所有最高票数并等概率随机抽一个
                            this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                  i[1] == this_cycle_semantic[0][1]]
                            this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                        # print(this_cycle_result)
                        simple_cycle_semantics_c.append(this_cycle_result)

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)

    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return d_rev, simple_cycles, simple_cycles_semantics


def get_cycle_basis_and_semantic_2_semansimplified(output_points, output_edges):
    # 一个关于索引和输出点的字典
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # 这里无法处理重复点，不能去掉nms
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # 这里无法处理重复点，不能去掉nms
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))


    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])
        G.add_edge(e[1], e[0])


    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []
    # print('断点1', simple_cycles)
    bridges = list(nx.bridges(G))
    # 虚边怎么做呢：还是直接删掉，从边集中删掉，然后作为多个连通的预测处理即可
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            G.remove_edge(b[1], b[0])
    # 断掉割边集以后，我们查看剩下的所有连通分量，此时只存在孤立点或圈，遍历剩下的所有圈
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            # 获取对应的点边集
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # 固定的边集，不会删除
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # 用于遍历以减少时间复杂度的边集，其中的边会删除
            # print(output_points_c)
            # print(output_edges_c)

            # 求该连通分量所有逆时针simple cycles的方法
            # 定义d中的编号为该连通分量中点的编号，
            # 遍历无向边集output_edges_c：
            # 对每条无向边，规定初始点为编号较小者，上一个点为初始点，当前点为编号较大者；
            # 求当前点的所有邻边，及对应的出射角度
            # 求上一个点入射到当前点的方向的反方向，对应的角度（最常见的极坐标系[0,2pi)）
            # 求从这个角度开始逆时针旋转，碰到的最后一个当前点的邻边
            # 将该邻边的另一端作为下一个点
            # 当下一个点等于初始点时，得到形式类似[p0,p1,...,pn-1,p0]的cycle
            # 检索cycle，从剩下的边中删除所有pi<pi+1的边（包括本边），遍历剩下的边
            # 将上一个点设为当前点，当前点设为下一个点

            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
                    try:
                        simple_cycle_semantics = []
                        simple_cycle = []
                        simple_cycle_number = []
                        point1 = edge_c[0]
                        point2 = edge_c[1]
                        point1_number = d[point1]
                        point2_number = d[point2]
                        # 初始点
                        initial_point = None
                        initial_point_number = None
                        if point1_number < point2_number:
                            initial_point = point1
                            initial_point_number = point1_number
                        else:
                            initial_point = point2
                            initial_point_number = point2_number
                        simple_cycle.append(initial_point)
                        simple_cycle_number.append(initial_point_number)
                        # 上一个点
                        last_point = initial_point
                        last_point_number = initial_point_number
                        # 当前点
                        current_point = None
                        current_point_number = None
                        if point1_number < point2_number:
                            current_point = point2
                            current_point_number = point2_number
                        else:
                            current_point = point1
                            current_point_number = point1_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)
                        # 初始点的后一个点（用于判断while结束）
                        next_initial_point = copy.deepcopy(current_point)
                        next_initial_point_number = copy.deepcopy(current_point_number)
                        # 下一个点
                        next_point = None
                        next_point_number = None
                        # 当下一个点等于初始点的后一个点时，结束
                        while_count = 0
                        while next_point != next_initial_point and while_count < 1000:
                            # 求当前点的所有邻边
                            relevant_edges = []
                            for edge in output_edges_c:
                                if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                    relevant_edges.append(edge)
                            # 求当前点的所有邻边对应的出射角度
                            relevant_edges_degree = []
                            for relevant_edge in relevant_edges:
                                # 出射向量
                                vec = None
                                if relevant_edge[0] == current_point:
                                    vec = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                elif relevant_edge[1] == current_point:
                                    vec = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                else:
                                    assert 0
                                # 求出射角度
                                vec_degree = x_axis_angle(vec)
                                relevant_edges_degree.append(vec_degree)
                            # 求上一个点入射到当前点的方向的反方向（出射方向）、对应出射角度
                            vec_from_current_point_to_last_point = None
                            vec_from_current_point_to_last_point_degree = None
                            for relevant_edge_ind, relevant_edge in enumerate(relevant_edges):
                                if relevant_edge == (current_point, last_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                elif relevant_edge == (last_point, current_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                else:
                                    continue
                            # 求从这个角度开始逆时针旋转，碰到的最后一个当前点的邻边
                            # 同时把没扫过部分（就是内角）的角度区域记录下来
                            # 这里我们的内角语义替换为完全语义
                            rotate_deltas_counterclockwise = []
                            # 记录内角区域，逆时针，从前一个角度到后一个角度
                            interior_angles = []
                            for relevant_edge_degree in relevant_edges_degree:
                                rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                    vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                                rotate_deltas_counterclockwise.append(rotate_delta)
                                interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                            # print(rotate_deltas_counterclockwise)
                            # 最大角对应索引
                            max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                            # 找到对应的内角
                            interior_angle_counterclockwise = interior_angles[max_rotate_index]
                            # 求出对应的语义区域
                            # 先求出当前点的所有语义，顺序按照四个象限排序
                            # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                            #                           current_point[4], ]
                            current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                      current_point[4], current_point[6], current_point[7],
                                                      current_point[8]]
                            # 求出该逆时针角占了四个象限的多少角度
                            # 求法：直接求度数较小的逆时针转到度数较大的覆盖四象限角度
                            # 然后判断，如果度数较小的是内角区域的“源方向”，则正好是覆盖四象限角度；
                            # 如果度数较小的是内角区域的“目标方向”，则对覆盖四象限角度用90度减去；
                            interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)  # 度数较小的
                            interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # 度数较大的
                            quadrant_smaller_to_bigger_counterclockwise = get_quadrant(
                                (interior_angle_counterclockwise_degree_smaller,
                                 interior_angle_counterclockwise_degree_bigger))
                            # print(quadrant_smaller_to_bigger_counterclockwise)
                            if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                                pass
                            elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                                quadrant_smaller_to_bigger_counterclockwise = (
                                90 - quadrant_smaller_to_bigger_counterclockwise[0],
                                90 - quadrant_smaller_to_bigger_counterclockwise[1],
                                90 - quadrant_smaller_to_bigger_counterclockwise[2],
                                90 - quadrant_smaller_to_bigger_counterclockwise[3])
                            else:
                                assert 0
                            # 永远不将语义置为-1
                            current_point_semantic_valid = []
                            for qd, seman in enumerate(current_point_semantic):
                                if 1:
                                    current_point_semantic_valid.append(seman)
                                else:
                                    current_point_semantic_valid.append(-1)
                            # 对全部语义进行统计
                            simple_cycle_semantics.append(current_point_semantic_valid)

                            # 对应边
                            max_rotate_edge = relevant_edges[max_rotate_index]
                            # 对应下一个点
                            if max_rotate_edge[0] == current_point:
                                next_point = max_rotate_edge[1]
                                next_point_number = d[next_point]
                            elif max_rotate_edge[1] == current_point:
                                next_point = max_rotate_edge[0]
                                next_point_number = d[next_point]
                            else:
                                assert 0
                            # 重新给上一个点、当前点、下一个点赋值，并将当前点加入simple_cycle
                            last_point = current_point
                            last_point_number = current_point_number
                            current_point = next_point
                            current_point_number = next_point_number
                            simple_cycle.append(current_point)
                            simple_cycle_number.append(current_point_number)
                            while_count += 1
                        if len(simple_cycle) > 800:
                            continue
                        # 最后加上初始点（为了删边）
                        # simple_cycle.append(initial_point)
                        # simple_cycle_number.append(initial_point_number)
                        # 检索simple_cycle_number，从剩下的边中删除所有pi<pi+1的边（包括本边）
                        # print('------------------')
                        # print(simple_cycle)
                        # print(simple_cycle_number)
                        # print('------------------')
                        for point_number_ind, point_number in enumerate(simple_cycle_number):
                            if point_number_ind < len(simple_cycle_number) - 1:
                                edge_number = (point_number, simple_cycle_number[point_number_ind + 1])
                                # print(simple_cycle_number)
                                if edge_number[0] < edge_number[1]:
                                    if (d_rev[edge_number[0]], d_rev[edge_number[1]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                    elif (
                                    d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[1]], d_rev[edge_number[0]]))
                        # 算面积时不需要闭环
                        simple_cycle.pop(-1)
                        simple_cycle_number.pop(-1)
                        # 存起来（逆时针计算面积，如果面积为负则不加入，说明是最大的那个）
                        polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                        polygon_counterclockwise.pop(-1)
                        # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                        if poly_area(polygon_counterclockwise) > 0:
                            simple_cycles_c.append(simple_cycle)
                            simple_cycles_number_c.append(simple_cycle_number)
                            # 公共最大语义（最大的那个圈就不用算了），得到该simple_cycle的语义并记录
                            semantic_result = {}
                            for semantic_label in range(0, 7):
                                semantic_result[semantic_label] = 0
                            for everypoint_semantic in simple_cycle_semantics:
                                for _ in range(0, 7):
                                    if _ in everypoint_semantic:
                                        semantic_result[_] += 1
                            del semantic_result[6]

                            # print(semantic_result)
                            # 如果最高票相同则等概率随机选一个（注意13不算）
                            this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                            # print(this_cycle_semantic)
                            this_cycle_result = None
                            if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                                # 以唯一最高票为准
                                this_cycle_result = this_cycle_semantic[0][0]
                            else:
                                # 找出所有最高票数并等概率随机抽一个
                                this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                      i[1] == this_cycle_semantic[0][1]]
                                this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                            # print(this_cycle_result)
                            simple_cycle_semantics_c.append(this_cycle_result)
                    except:
                        pass

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)

    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return d_rev, simple_cycles, simple_cycles_semantics

def get_cycle_basis_and_semantic_3_semansimplified(output_points, output_edges):
    # 一个关于索引和输出点的字典
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # 这里无法处理重复点，不能去掉nms
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # 这里无法处理重复点，不能去掉nms
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))


    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])
        G.add_edge(e[1], e[0])


    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []
    # print('断点1', simple_cycles)
    bridges = list(nx.bridges(G))
    # 虚边怎么做呢：还是直接删掉，从边集中删掉，然后作为多个连通的预测处理即可
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            if G.has_edge(b[0], b[1]):
                G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            if G.has_edge(b[1], b[0]):
                G.remove_edge(b[1], b[0])
    # 断掉割边集以后，我们查看剩下的所有连通分量，此时只存在孤立点或圈，遍历剩下的所有圈
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            # 获取对应的点边集
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # 固定的边集，不会删除
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # 用于遍历以减少时间复杂度的边集，其中的边会删除
            # print(output_points_c)
            # print(output_edges_c)

            # 求该连通分量所有逆时针simple cycles的方法
            # 定义d中的编号为该连通分量中点的编号，
            # 遍历无向边集output_edges_c：
            # 对每条无向边，规定初始点为编号较小者，上一个点为初始点，当前点为编号较大者；
            # 求当前点的所有邻边，及对应的出射角度
            # 求上一个点入射到当前点的方向的反方向，对应的角度（最常见的极坐标系[0,2pi)）
            # 求从这个角度开始逆时针旋转，碰到的最后一个当前点的邻边
            # 将该邻边的另一端作为下一个点
            # 当下一个点等于初始点时，得到形式类似[p0,p1,...,pn-1,p0]的cycle
            # 检索cycle，从剩下的边中删除所有pi<pi+1的边（包括本边），遍历剩下的边
            # 将上一个点设为当前点，当前点设为下一个点

            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
                    try:
                        simple_cycle_semantics = []
                        simple_cycle = []
                        simple_cycle_number = []
                        point1 = edge_c[0]
                        point2 = edge_c[1]
                        point1_number = d[point1]
                        point2_number = d[point2]
                        # 初始点
                        initial_point = None
                        initial_point_number = None
                        if point1_number < point2_number:
                            initial_point = point1
                            initial_point_number = point1_number
                        else:
                            initial_point = point2
                            initial_point_number = point2_number
                        simple_cycle.append(initial_point)
                        simple_cycle_number.append(initial_point_number)
                        # 上一个点
                        last_point = initial_point
                        last_point_number = initial_point_number
                        # 当前点
                        current_point = None
                        current_point_number = None
                        if point1_number < point2_number:
                            current_point = point2
                            current_point_number = point2_number
                        else:
                            current_point = point1
                            current_point_number = point1_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)
                        # 初始点的后一个点（用于判断while结束）
                        next_initial_point = copy.deepcopy(current_point)
                        next_initial_point_number = copy.deepcopy(current_point_number)
                        # 下一个点
                        next_point = None
                        next_point_number = None
                        # 当下一个点等于初始点的后一个点时，结束
                        while_count = 0
                        while next_point != next_initial_point and while_count < 100:
                            # 求当前点的所有邻边
                            relevant_edges = []
                            for edge in output_edges_c:
                                if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                    relevant_edges.append(edge)
                            # 求当前点的所有邻边对应的出射角度
                            relevant_edges_degree = []
                            for relevant_edge in relevant_edges:
                                # 出射向量
                                vec = None
                                if relevant_edge[0] == current_point:
                                    vec = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                elif relevant_edge[1] == current_point:
                                    vec = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                else:
                                    assert 0
                                # 求出射角度
                                vec_degree = x_axis_angle(vec)
                                relevant_edges_degree.append(vec_degree)
                            # 求上一个点入射到当前点的方向的反方向（出射方向）、对应出射角度
                            vec_from_current_point_to_last_point = None
                            vec_from_current_point_to_last_point_degree = None
                            for relevant_edge_ind, relevant_edge in enumerate(relevant_edges):
                                if relevant_edge == (current_point, last_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                elif relevant_edge == (last_point, current_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                else:
                                    continue
                            # 求从这个角度开始逆时针旋转，碰到的最后一个当前点的邻边
                            # 同时把没扫过部分（就是内角）的角度区域记录下来
                            # 这里我们的内角语义替换为完全语义
                            rotate_deltas_counterclockwise = []
                            # 记录内角区域，逆时针，从前一个角度到后一个角度
                            interior_angles = []
                            for relevant_edge_degree in relevant_edges_degree:
                                rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                    vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                                rotate_deltas_counterclockwise.append(rotate_delta)
                                interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                            # print(rotate_deltas_counterclockwise)
                            # 最大角对应索引
                            max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                            # 找到对应的内角
                            interior_angle_counterclockwise = interior_angles[max_rotate_index]
                            # 求出对应的语义区域
                            # 先求出当前点的所有语义，顺序按照四个象限排序
                            # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                            #                           current_point[4], ]
                            current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                      current_point[4], current_point[6], current_point[7],
                                                      current_point[8]]
                            # 求出该逆时针角占了四个象限的多少角度
                            # 求法：直接求度数较小的逆时针转到度数较大的覆盖四象限角度
                            # 然后判断，如果度数较小的是内角区域的“源方向”，则正好是覆盖四象限角度；
                            # 如果度数较小的是内角区域的“目标方向”，则对覆盖四象限角度用90度减去；
                            interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)  # 度数较小的
                            interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # 度数较大的
                            quadrant_smaller_to_bigger_counterclockwise = get_quadrant(
                                (interior_angle_counterclockwise_degree_smaller,
                                 interior_angle_counterclockwise_degree_bigger))
                            # print(quadrant_smaller_to_bigger_counterclockwise)
                            if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                                pass
                            elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                                quadrant_smaller_to_bigger_counterclockwise = (
                                90 - quadrant_smaller_to_bigger_counterclockwise[0],
                                90 - quadrant_smaller_to_bigger_counterclockwise[1],
                                90 - quadrant_smaller_to_bigger_counterclockwise[2],
                                90 - quadrant_smaller_to_bigger_counterclockwise[3])
                            else:
                                assert 0
                            # 永远不将语义置为-1
                            current_point_semantic_valid = []
                            for qd, seman in enumerate(current_point_semantic):
                                if 1:
                                    current_point_semantic_valid.append(seman)
                                else:
                                    current_point_semantic_valid.append(-1)
                            # 对全部语义进行统计
                            simple_cycle_semantics.append(current_point_semantic_valid)

                            # 对应边
                            max_rotate_edge = relevant_edges[max_rotate_index]
                            # 对应下一个点
                            if max_rotate_edge[0] == current_point:
                                next_point = max_rotate_edge[1]
                                next_point_number = d[next_point]
                            elif max_rotate_edge[1] == current_point:
                                next_point = max_rotate_edge[0]
                                next_point_number = d[next_point]
                            else:
                                assert 0
                            # 重新给上一个点、当前点、下一个点赋值，并将当前点加入simple_cycle
                            last_point = current_point
                            last_point_number = current_point_number
                            current_point = next_point
                            current_point_number = next_point_number
                            simple_cycle.append(current_point)
                            simple_cycle_number.append(current_point_number)
                            while_count += 1
                        if len(simple_cycle) > 80:
                            continue
                        # 最后加上初始点（为了删边）
                        # simple_cycle.append(initial_point)
                        # simple_cycle_number.append(initial_point_number)
                        # 检索simple_cycle_number，从剩下的边中删除所有pi<pi+1的边（包括本边）
                        # print('------------------')
                        # print(simple_cycle)
                        # print(simple_cycle_number)
                        # print('------------------')
                        for point_number_ind, point_number in enumerate(simple_cycle_number):
                            if point_number_ind < len(simple_cycle_number) - 1:
                                edge_number = (point_number, simple_cycle_number[point_number_ind + 1])
                                # print(simple_cycle_number)
                                if edge_number[0] < edge_number[1]:
                                    if (d_rev[edge_number[0]], d_rev[edge_number[1]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                    elif (
                                    d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[1]], d_rev[edge_number[0]]))
                        # 算面积时不需要闭环
                        simple_cycle.pop(-1)
                        simple_cycle_number.pop(-1)
                        # 存起来（逆时针计算面积，如果面积为负则不加入，说明是最大的那个）
                        polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                        polygon_counterclockwise.pop(-1)
                        # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                        if poly_area(polygon_counterclockwise) > 0:
                            simple_cycles_c.append(simple_cycle)
                            simple_cycles_number_c.append(simple_cycle_number)
                            # 公共最大语义（最大的那个圈就不用算了），得到该simple_cycle的语义并记录
                            semantic_result = {}
                            for semantic_label in range(0, 7):
                                semantic_result[semantic_label] = 0
                            for everypoint_semantic in simple_cycle_semantics:
                                for _ in range(0, 7):
                                    if _ in everypoint_semantic:
                                        semantic_result[_] += 1
                            del semantic_result[6]

                            # print(semantic_result)
                            # 如果最高票相同则等概率随机选一个（注意13不算）
                            this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                            # print(this_cycle_semantic)
                            if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                                # 以唯一最高票为准
                                this_cycle_result = this_cycle_semantic[0][0]
                            else:
                                # 找出所有最高票数，按照橱柜2，浴室4，厨房3，卧室1，阳台5，客厅0的优先级确定
                                this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                      i[1] == this_cycle_semantic[0][1]]
                                if 2 in this_cycle_results:
                                    this_cycle_result = 2
                                elif 4 in this_cycle_results:
                                    this_cycle_result = 4
                                elif 3 in this_cycle_results:
                                    this_cycle_result = 3
                                elif 1 in this_cycle_results:
                                    this_cycle_result = 1
                                elif 5 in this_cycle_results:
                                    this_cycle_result = 5
                                else:
                                    this_cycle_result = 0
                            # print(this_cycle_result)
                            simple_cycle_semantics_c.append(this_cycle_result)
                    except:
                        pass

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)

    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return d_rev, simple_cycles, simple_cycles_semantics

def get_cycle_basis_and_semantic_3_semansimplified_lifull(output_points, output_edges):
    # 一个关于索引和输出点的字典
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # 这里无法处理重复点，不能去掉nms
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # 这里无法处理重复点，不能去掉nms
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))


    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])
        G.add_edge(e[1], e[0])


    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []
    # print('断点1', simple_cycles)
    bridges = list(nx.bridges(G))
    # 虚边怎么做呢：还是直接删掉，从边集中删掉，然后作为多个连通的预测处理即可
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            if G.has_edge(b[0], b[1]):
                G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            if G.has_edge(b[1], b[0]):
                G.remove_edge(b[1], b[0])
    # 断掉割边集以后，我们查看剩下的所有连通分量，此时只存在孤立点或圈，遍历剩下的所有圈
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # 获取对应的点边集
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # 固定的边集，不会删除
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # 用于遍历以减少时间复杂度的边集，其中的边会删除


            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
                    try:
                        simple_cycle_semantics = []
                        simple_cycle = []
                        simple_cycle_number = []
                        point1 = edge_c[0]
                        point2 = edge_c[1]
                        point1_number = d[point1]
                        point2_number = d[point2]
                        # 初始点
                        initial_point = None
                        initial_point_number = None
                        if point1_number < point2_number:
                            initial_point = point1
                            initial_point_number = point1_number
                        else:
                            initial_point = point2
                            initial_point_number = point2_number
                        simple_cycle.append(initial_point)
                        simple_cycle_number.append(initial_point_number)
                        # 上一个点
                        last_point = initial_point
                        last_point_number = initial_point_number
                        # 当前点
                        current_point = None
                        current_point_number = None
                        if point1_number < point2_number:
                            current_point = point2
                            current_point_number = point2_number
                        else:
                            current_point = point1
                            current_point_number = point1_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)
                        # 初始点的后一个点（用于判断while结束）
                        next_initial_point = copy.deepcopy(current_point)
                        next_initial_point_number = copy.deepcopy(current_point_number)
                        # 下一个点
                        next_point = None
                        next_point_number = None
                        # 当下一个点等于初始点的后一个点时，结束
                        while_count = 0
                        while next_point != next_initial_point and while_count < 100:
                            # 求当前点的所有邻边
                            relevant_edges = []
                            for edge in output_edges_c:
                                if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                    relevant_edges.append(edge)
                            # 求当前点的所有邻边对应的出射角度
                            relevant_edges_degree = []
                            for relevant_edge in relevant_edges:
                                # 出射向量
                                vec = None
                                if relevant_edge[0] == current_point:
                                    vec = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                elif relevant_edge[1] == current_point:
                                    vec = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                else:
                                    assert 0
                                # 求出射角度
                                vec_degree = x_axis_angle(vec)
                                relevant_edges_degree.append(vec_degree)
                            # 求上一个点入射到当前点的方向的反方向（出射方向）、对应出射角度
                            vec_from_current_point_to_last_point = None
                            vec_from_current_point_to_last_point_degree = None
                            for relevant_edge_ind, relevant_edge in enumerate(relevant_edges):
                                if relevant_edge == (current_point, last_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                elif relevant_edge == (last_point, current_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                else:
                                    continue
                            # 求从这个角度开始逆时针旋转，碰到的最后一个当前点的邻边
                            # 同时把没扫过部分（就是内角）的角度区域记录下来
                            # 这里我们的内角语义替换为完全语义
                            rotate_deltas_counterclockwise = []
                            # 记录内角区域，逆时针，从前一个角度到后一个角度
                            interior_angles = []
                            for relevant_edge_degree in relevant_edges_degree:
                                rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                    vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                                rotate_deltas_counterclockwise.append(rotate_delta)
                                interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                            # print(rotate_deltas_counterclockwise)
                            # 最大角对应索引
                            max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                            # 找到对应的内角
                            interior_angle_counterclockwise = interior_angles[max_rotate_index]
                            # 求出对应的语义区域
                            # 先求出当前点的所有语义，顺序按照四个象限排序
                            # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                            #                           current_point[4], ]
                            current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                      current_point[4], current_point[6], current_point[7],
                                                      current_point[8], current_point[9], current_point[10], current_point[11], current_point[12], current_point[13], current_point[14]]
                            # 求出该逆时针角占了四个象限的多少角度
                            # 求法：直接求度数较小的逆时针转到度数较大的覆盖四象限角度
                            # 然后判断，如果度数较小的是内角区域的“源方向”，则正好是覆盖四象限角度；
                            # 如果度数较小的是内角区域的“目标方向”，则对覆盖四象限角度用90度减去；
                            interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)  # 度数较小的
                            interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # 度数较大的
                            quadrant_smaller_to_bigger_counterclockwise = get_quadrant(
                                (interior_angle_counterclockwise_degree_smaller,
                                 interior_angle_counterclockwise_degree_bigger))
                            # print(quadrant_smaller_to_bigger_counterclockwise)
                            if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                                pass
                            elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                                quadrant_smaller_to_bigger_counterclockwise = (
                                90 - quadrant_smaller_to_bigger_counterclockwise[0],
                                90 - quadrant_smaller_to_bigger_counterclockwise[1],
                                90 - quadrant_smaller_to_bigger_counterclockwise[2],
                                90 - quadrant_smaller_to_bigger_counterclockwise[3])
                            else:
                                assert 0
                            # 永远不将语义置为-1
                            current_point_semantic_valid = []
                            for qd, seman in enumerate(current_point_semantic):
                                if 1:
                                    current_point_semantic_valid.append(seman)
                                else:
                                    current_point_semantic_valid.append(-1)
                            # 对全部语义进行统计
                            simple_cycle_semantics.append(current_point_semantic_valid)

                            # 对应边
                            max_rotate_edge = relevant_edges[max_rotate_index]
                            # 对应下一个点
                            if max_rotate_edge[0] == current_point:
                                next_point = max_rotate_edge[1]
                                next_point_number = d[next_point]
                            elif max_rotate_edge[1] == current_point:
                                next_point = max_rotate_edge[0]
                                next_point_number = d[next_point]
                            else:
                                assert 0
                            # 重新给上一个点、当前点、下一个点赋值，并将当前点加入simple_cycle
                            last_point = current_point
                            last_point_number = current_point_number
                            current_point = next_point
                            current_point_number = next_point_number
                            simple_cycle.append(current_point)
                            simple_cycle_number.append(current_point_number)
                            while_count += 1
                        if len(simple_cycle) > 80:
                            continue
                        # 最后加上初始点（为了删边）
                        # simple_cycle.append(initial_point)
                        # simple_cycle_number.append(initial_point_number)
                        # 检索simple_cycle_number，从剩下的边中删除所有pi<pi+1的边（包括本边）
                        # print('------------------')
                        # print(simple_cycle)
                        # print(simple_cycle_number)
                        # print('------------------')
                        for point_number_ind, point_number in enumerate(simple_cycle_number):
                            if point_number_ind < len(simple_cycle_number) - 1:
                                edge_number = (point_number, simple_cycle_number[point_number_ind + 1])
                                # print(simple_cycle_number)
                                if edge_number[0] < edge_number[1]:
                                    if (d_rev[edge_number[0]], d_rev[edge_number[1]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                    elif (
                                    d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[1]], d_rev[edge_number[0]]))
                        # 算面积时不需要闭环
                        simple_cycle.pop(-1)
                        simple_cycle_number.pop(-1)
                        # 存起来（逆时针计算面积，如果面积为负则不加入，说明是最大的那个）
                        polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                        polygon_counterclockwise.pop(-1)
                        # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                        if poly_area(polygon_counterclockwise) > 0:
                            simple_cycles_c.append(simple_cycle)
                            simple_cycles_number_c.append(simple_cycle_number)
                            # 公共最大语义（最大的那个圈就不用算了），得到该simple_cycle的语义并记录
                            semantic_result = {}
                            for semantic_label in range(0, 13):
                                semantic_result[semantic_label] = 0
                            for everypoint_semantic in simple_cycle_semantics:
                                for _ in range(0, 13):
                                    if _ in everypoint_semantic:
                                        semantic_result[_] += 1
                            del semantic_result[11]
                            del semantic_result[12]

                            # print(semantic_result)
                            # 如果最高票相同则等概率随机选一个（注意13不算）
                            this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                            # print(this_cycle_semantic)
                            if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                                # 以唯一最高票为准
                                this_cycle_result = this_cycle_semantic[0][0]
                            else:
                                # 找出所有最高票数，按照0 10 1 8 9 5 6 2 4 3 7的优先级确定，这个不好观察，直接简单按频率
                                this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                      i[1] == this_cycle_semantic[0][1]]
                                if 0 in this_cycle_results:
                                    this_cycle_result = 0
                                elif 10 in this_cycle_results:
                                    this_cycle_result = 10
                                elif 1 in this_cycle_results:
                                    this_cycle_result = 1
                                elif 8 in this_cycle_results:
                                    this_cycle_result = 8
                                elif 9 in this_cycle_results:
                                    this_cycle_result = 9
                                elif 5 in this_cycle_results:
                                    this_cycle_result = 5
                                elif 6 in this_cycle_results:
                                    this_cycle_result = 6
                                elif 2 in this_cycle_results:
                                    this_cycle_result = 2
                                elif 4 in this_cycle_results:
                                    this_cycle_result = 4
                                elif 3 in this_cycle_results:
                                    this_cycle_result = 3
                                else:
                                    this_cycle_result = 7
                            # print(this_cycle_result)
                            simple_cycle_semantics_c.append(this_cycle_result)
                    except:
                        pass

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)

    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return d_rev, simple_cycles, simple_cycles_semantics

def get_cycle_basis_and_semantic_2_semansimplified_4extractingboundary(output_points, output_edges):
    # 一个关于索引和输出点的字典
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # 这里无法处理重复点，不能去掉nms
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # 这里无法处理重复点，不能去掉nms
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))


    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])
        G.add_edge(e[1], e[0])


    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []
    # print('断点1', simple_cycles)
    bridges = list(nx.bridges(G))
    # 虚边怎么做呢：还是直接删掉，从边集中删掉，然后作为多个连通的预测处理即可
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            G.remove_edge(b[1], b[0])
    # 断掉割边集以后，我们查看剩下的所有连通分量，此时只存在孤立点或圈，遍历剩下的所有圈
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            # 获取对应的点边集
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # 固定的边集，不会删除
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # 用于遍历以减少时间复杂度的边集，其中的边会删除
            # print(output_points_c)
            # print(output_edges_c)

            # 求该连通分量所有逆时针simple cycles的方法
            # 定义d中的编号为该连通分量中点的编号，
            # 遍历无向边集output_edges_c：
            # 对每条无向边，规定初始点为编号较小者，上一个点为初始点，当前点为编号较大者；
            # 求当前点的所有邻边，及对应的出射角度
            # 求上一个点入射到当前点的方向的反方向，对应的角度（最常见的极坐标系[0,2pi)）
            # 求从这个角度开始逆时针旋转，碰到的最后一个当前点的邻边
            # 将该邻边的另一端作为下一个点
            # 当下一个点等于初始点时，得到形式类似[p0,p1,...,pn-1,p0]的cycle
            # 检索cycle，从剩下的边中删除所有pi<pi+1的边（包括本边），遍历剩下的边
            # 将上一个点设为当前点，当前点设为下一个点

            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
                    try:
                        simple_cycle_semantics = []
                        simple_cycle = []
                        simple_cycle_number = []
                        point1 = edge_c[0]
                        point2 = edge_c[1]
                        point1_number = d[point1]
                        point2_number = d[point2]
                        # 初始点
                        initial_point = None
                        initial_point_number = None
                        if point1_number < point2_number:
                            initial_point = point1
                            initial_point_number = point1_number
                        else:
                            initial_point = point2
                            initial_point_number = point2_number
                        simple_cycle.append(initial_point)
                        simple_cycle_number.append(initial_point_number)
                        # 上一个点
                        last_point = initial_point
                        last_point_number = initial_point_number
                        # 当前点
                        current_point = None
                        current_point_number = None
                        if point1_number < point2_number:
                            current_point = point2
                            current_point_number = point2_number
                        else:
                            current_point = point1
                            current_point_number = point1_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)
                        # 初始点的后一个点（用于判断while结束）
                        next_initial_point = copy.deepcopy(current_point)
                        next_initial_point_number = copy.deepcopy(current_point_number)
                        # 下一个点
                        next_point = None
                        next_point_number = None
                        # 当下一个点等于初始点的后一个点时，结束
                        while_count = 0
                        while next_point != next_initial_point and while_count < 1000:
                            # 求当前点的所有邻边
                            relevant_edges = []
                            for edge in output_edges_c:
                                if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                    relevant_edges.append(edge)
                            # 求当前点的所有邻边对应的出射角度
                            relevant_edges_degree = []
                            for relevant_edge in relevant_edges:
                                # 出射向量
                                vec = None
                                if relevant_edge[0] == current_point:
                                    vec = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                elif relevant_edge[1] == current_point:
                                    vec = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                else:
                                    assert 0
                                # 求出射角度
                                vec_degree = x_axis_angle(vec)
                                relevant_edges_degree.append(vec_degree)
                            # 求上一个点入射到当前点的方向的反方向（出射方向）、对应出射角度
                            vec_from_current_point_to_last_point = None
                            vec_from_current_point_to_last_point_degree = None
                            for relevant_edge_ind, relevant_edge in enumerate(relevant_edges):
                                if relevant_edge == (current_point, last_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                elif relevant_edge == (last_point, current_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                else:
                                    continue
                            # 求从这个角度开始逆时针旋转，碰到的最后一个当前点的邻边
                            # 同时把没扫过部分（就是内角）的角度区域记录下来
                            # 这里我们的内角语义替换为完全语义
                            rotate_deltas_counterclockwise = []
                            # 记录内角区域，逆时针，从前一个角度到后一个角度
                            interior_angles = []
                            for relevant_edge_degree in relevant_edges_degree:
                                rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                    vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                                rotate_deltas_counterclockwise.append(rotate_delta)
                                interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                            # print(rotate_deltas_counterclockwise)
                            # 最大角对应索引
                            max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                            # 找到对应的内角
                            interior_angle_counterclockwise = interior_angles[max_rotate_index]
                            # 求出对应的语义区域
                            # 先求出当前点的所有语义，顺序按照四个象限排序
                            # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                            #                           current_point[4], ]
                            current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                      current_point[4], current_point[6], current_point[7],
                                                      current_point[8]]
                            # 求出该逆时针角占了四个象限的多少角度
                            # 求法：直接求度数较小的逆时针转到度数较大的覆盖四象限角度
                            # 然后判断，如果度数较小的是内角区域的“源方向”，则正好是覆盖四象限角度；
                            # 如果度数较小的是内角区域的“目标方向”，则对覆盖四象限角度用90度减去；
                            interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)  # 度数较小的
                            interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # 度数较大的
                            quadrant_smaller_to_bigger_counterclockwise = get_quadrant(
                                (interior_angle_counterclockwise_degree_smaller,
                                 interior_angle_counterclockwise_degree_bigger))
                            # print(quadrant_smaller_to_bigger_counterclockwise)
                            if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                                pass
                            elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                                quadrant_smaller_to_bigger_counterclockwise = (
                                90 - quadrant_smaller_to_bigger_counterclockwise[0],
                                90 - quadrant_smaller_to_bigger_counterclockwise[1],
                                90 - quadrant_smaller_to_bigger_counterclockwise[2],
                                90 - quadrant_smaller_to_bigger_counterclockwise[3])
                            else:
                                assert 0
                            # 永远不将语义置为-1
                            current_point_semantic_valid = []
                            for qd, seman in enumerate(current_point_semantic):
                                if 1:
                                    current_point_semantic_valid.append(seman)
                                else:
                                    current_point_semantic_valid.append(-1)
                            # 对全部语义进行统计
                            simple_cycle_semantics.append(current_point_semantic_valid)

                            # 对应边
                            max_rotate_edge = relevant_edges[max_rotate_index]
                            # 对应下一个点
                            if max_rotate_edge[0] == current_point:
                                next_point = max_rotate_edge[1]
                                next_point_number = d[next_point]
                            elif max_rotate_edge[1] == current_point:
                                next_point = max_rotate_edge[0]
                                next_point_number = d[next_point]
                            else:
                                assert 0
                            # 重新给上一个点、当前点、下一个点赋值，并将当前点加入simple_cycle
                            last_point = current_point
                            last_point_number = current_point_number
                            current_point = next_point
                            current_point_number = next_point_number
                            simple_cycle.append(current_point)
                            simple_cycle_number.append(current_point_number)
                            while_count += 1
                        if len(simple_cycle) > 800:
                            continue
                        # 最后加上初始点（为了删边）
                        # simple_cycle.append(initial_point)
                        # simple_cycle_number.append(initial_point_number)
                        # 检索simple_cycle_number，从剩下的边中删除所有pi<pi+1的边（包括本边）
                        # print('------------------')
                        # print(simple_cycle)
                        # print(simple_cycle_number)
                        # print('------------------')
                        for point_number_ind, point_number in enumerate(simple_cycle_number):
                            if point_number_ind < len(simple_cycle_number) - 1:
                                edge_number = (point_number, simple_cycle_number[point_number_ind + 1])
                                # print(simple_cycle_number)
                                if edge_number[0] < edge_number[1]:
                                    if (d_rev[edge_number[0]], d_rev[edge_number[1]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                    elif (
                                    d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[1]], d_rev[edge_number[0]]))
                        # 算面积时不需要闭环
                        simple_cycle.pop(-1)
                        simple_cycle_number.pop(-1)
                        # 存起来（逆时针计算面积，如果面积为负则不加入，说明是最大的那个）
                        polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                        polygon_counterclockwise.pop(-1)
                        # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))

                        # if poly_area(polygon_counterclockwise) > 0:
                        if 1:
                            simple_cycles_c.append(simple_cycle)
                            simple_cycles_number_c.append(simple_cycle_number)
                            # 公共最大语义（最大的那个圈就不用算了），得到该simple_cycle的语义并记录
                            semantic_result = {}
                            for semantic_label in range(0, 7):
                                semantic_result[semantic_label] = 0
                            for everypoint_semantic in simple_cycle_semantics:
                                for _ in range(0, 7):
                                    if _ in everypoint_semantic:
                                        semantic_result[_] += 1

                            # print(semantic_result)
                            # 如果最高票相同则等概率随机选一个（注意13不算）
                            this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                            this_cycle_result = None
                            if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                                # 以唯一最高票为准
                                this_cycle_result = this_cycle_semantic[0][0]
                            else:
                                # 找出所有最高票数并等概率随机抽一个
                                this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                      i[1] == this_cycle_semantic[0][1]]
                                if 6 in this_cycle_results:
                                    if poly_area(polygon_counterclockwise) > 0:
                                        this_cycle_results.remove(6)
                                        this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                                    else:
                                        this_cycle_result = 6
                                else:
                                    this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                            # print(this_cycle_result)
                            simple_cycle_semantics_c.append(this_cycle_result)
                    except:
                        pass

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)

    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return d_rev, simple_cycles, simple_cycles_semantics

def edges_to_coordinates(edges_array, vertices_list):
    n = int(np.sqrt(len(edges_array)))  # 根据边数组长度计算顶点数目
    edges_coordinates = []
    # 遍历边数组，找到值为 1 的位置
    for idx, value in enumerate(edges_array):
        if value == 1:
            # 计算对应的顶点索引
            vertex1_idx, vertex2_idx = divmod(idx, n)
            # 从顶点列表中获取对应的坐标
            vertex1 = vertices_list[vertex1_idx]
            vertex2 = vertices_list[vertex2_idx]
            # 将边表示为坐标对
            edge_coordinates = (vertex1, vertex2)
            edges_coordinates.append(edge_coordinates)
    return edges_coordinates

# merge points
def get_near_corners(points_array, merge_threshold):
    points = points_array.reshape(-1, 2)
    n = len(points)
    distance_matrix = np.ones((n, n)) * 999999
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.max(np.abs(points[i] - points[j]))
    edges = np.argwhere(distance_matrix < merge_threshold)
    edges = edges[edges[:, 0] != edges[:, 1]]
    graph = nx.Graph()
    graph.add_edges_from(edges)
    components = list(nx.connected_components(graph))
    return components

def merge_array_elements(array, full_indices_list, random_indices_list):
    merged_elements = []
    for i in range(len(array)):
        if i in full_indices_list and i not in random_indices_list:
            continue
        else:
            merged_elements.append(array[i])
    return np.array(merged_elements)

def graph_level_feature_analysis_for_cvae():
    import pandas as pd

    s0n = []
    s1n = []
    s2n = []
    s3n = []
    s4n = []
    s5n = []
    cn = []
    rn = []
    en = []
    ln = []
    test_files = os.listdir('../datasets/rplang-v3-bubble-diagram/test')
    for test_file in tqdm(test_files):
        test_graph = np.load('../datasets/rplang-v3-bubble-diagram/test/' + test_file, allow_pickle=True).item()

        s0n.append(test_graph['semantics'].count(0))
        s1n.append(test_graph['semantics'].count(1))
        s2n.append(test_graph['semantics'].count(2))
        s3n.append(test_graph['semantics'].count(3))
        s4n.append(test_graph['semantics'].count(4))
        s5n.append(test_graph['semantics'].count(5))

        cn.append(test_graph['corner_number'])
        rn.append(len(test_graph['polygons']))
        en.append(np.sum(np.triu(np.array(test_graph['adjacency_matrix']))))

        # 找到所有圈的数量
        circles_of_length_3 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 3)) / 6
        circles_of_length_4 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 4)) / 8
        circles_of_length_5 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 5)) / 10
        circles_of_length_6 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 6)) / 12
        circles_of_length_7 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 7)) / 14
        circles_of_length_8 = np.trace(np.linalg.matrix_power(np.array(test_graph['adjacency_matrix']), 8)) / 16
        # loopn = circles_of_length_3 + circles_of_length_4 + circles_of_length_5 + circles_of_length_6 + circles_of_length_7 + circles_of_length_8
        loopn = circles_of_length_3
        # print(len(test_graph['polygons']))
        # print(circles_of_length_3)
        # print(circles_of_length_4)
        # print(circles_of_length_5)
        # print(circles_of_length_6)
        # print(circles_of_length_7)
        # print(circles_of_length_8)

        ln.append(loopn)

    train_files = os.listdir('../datasets/rplang-v3-bubble-diagram/train')
    for train_file in tqdm(train_files):
        train_graph = np.load('../datasets/rplang-v3-bubble-diagram/train/' + train_file, allow_pickle=True).item()

        s0n.append(train_graph['semantics'].count(0))
        s1n.append(train_graph['semantics'].count(1))
        s2n.append(train_graph['semantics'].count(2))
        s3n.append(train_graph['semantics'].count(3))
        s4n.append(train_graph['semantics'].count(4))
        s5n.append(train_graph['semantics'].count(5))

        cn.append(train_graph['corner_number'])
        rn.append(len(train_graph['polygons']))
        en.append(np.sum(np.triu(np.array(train_graph['adjacency_matrix']))))

        # 找到所有圈的数量
        circles_of_length_3 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 3)) / 6
        circles_of_length_4 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 4)) / 8
        circles_of_length_5 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 5)) / 10
        circles_of_length_6 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 6)) / 12
        circles_of_length_7 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 7)) / 14
        circles_of_length_8 = np.trace(np.linalg.matrix_power(np.array(train_graph['adjacency_matrix']), 8)) / 16
        # loopn = circles_of_length_3 + circles_of_length_4 + circles_of_length_5 + circles_of_length_6 + circles_of_length_7 + circles_of_length_8
        loopn = circles_of_length_3
        ln.append(loopn)

    val_files = os.listdir('../datasets/rplang-v3-bubble-diagram/val')
    for val_file in tqdm(val_files):
        val_graph = np.load('../datasets/rplang-v3-bubble-diagram/val/' + val_file, allow_pickle=True).item()

        s0n.append(val_graph['semantics'].count(0))
        s1n.append(val_graph['semantics'].count(1))
        s2n.append(val_graph['semantics'].count(2))
        s3n.append(val_graph['semantics'].count(3))
        s4n.append(val_graph['semantics'].count(4))
        s5n.append(val_graph['semantics'].count(5))

        cn.append(val_graph['corner_number'])
        rn.append(len(val_graph['polygons']))
        en.append(np.sum(np.triu(np.array(val_graph['adjacency_matrix']))))

        # 找到所有圈的数量
        circles_of_length_3 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 3)) / 6
        circles_of_length_4 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 4)) / 8
        circles_of_length_5 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 5)) / 10
        circles_of_length_6 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 6)) / 12
        circles_of_length_7 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 7)) / 14
        circles_of_length_8 = np.trace(np.linalg.matrix_power(np.array(val_graph['adjacency_matrix']), 8)) / 16
        # loopn = circles_of_length_3 + circles_of_length_4 + circles_of_length_5 + circles_of_length_6 + circles_of_length_7 + circles_of_length_8
        loopn = circles_of_length_3
        ln.append(loopn)

    df = pd.DataFrame(
        {
            'corner_number': cn,
            'room_number': rn,
            'edge_number': en,
            'living_room_number': s0n,
            'bedroom_number': s1n,
            'kitchen_number': s2n,
            'bathroom_number': s3n,
            'balcony_number': s4n,
            'storage_number': s5n,
            'loop_number': ln,
        }
    )
    # 计算相关性矩阵
    correlation_matrix = df.corr()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('--------------------------------------------------------------------------------------------------')
        print(correlation_matrix)

    # 计算斯皮尔曼秩相关系数
    spearman_corr_matrix = df.corr(method='spearman')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('--------------------------------------------------------------------------------------------------')
        print(spearman_corr_matrix)

    # 计算肯德尔等级相关系数
    kendall_corr_matrix = df.corr(method='kendall')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('--------------------------------------------------------------------------------------------------')
        print(kendall_corr_matrix)
    print(ln)
    assert 0