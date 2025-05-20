import copy
import math
import os, json, numpy as np, cv2, random
from collections import Counter
import networkx as nx
from tqdm import *

image_path = r'/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/rplan_dataset/floorplan_dataset'
# print(len(os.listdir(image_path)))

train = [int(fn[:-4]) for fn in os.listdir(r'/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-withsemantics/train')]
test = [int(fn[:-4]) for fn in os.listdir(r'/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-withsemantics/test')]


def get_label(data):
    # 提取整数标签
    integer_labels = [item[1] for item in data if item[1] <= 12]
    # 统计每个整数标签出现的次数
    label_counts = Counter(integer_labels)
    # 找到出现次数最多的整数标签
    max_label = max(label_counts, key=label_counts.get)
    return max_label


def get_points_and_pixel_values_inside_polygon(image, polygon_vertices):
    # 计算多边形的边界框
    min_x = min([vertex[0] for vertex in polygon_vertices])
    max_x = max([vertex[0] for vertex in polygon_vertices])
    min_y = min([vertex[1] for vertex in polygon_vertices])
    max_y = max([vertex[1] for vertex in polygon_vertices])
    # 构造一个空白图像（与多边形的边界框尺寸相同）
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    blank_image = np.zeros((height, width), dtype=np.uint8)
    # 为多边形点创建numpy数组
    polygon_np = np.array([polygon_vertices], dtype=np.int32)
    # 填充多边形
    mask = cv2.fillPoly(blank_image, polygon_np - [min_x, min_y], 255)
    # 获取多边形内部的所有点
    points_inside_polygon = np.argwhere(mask == 255)
    # 将点坐标转换回原始图像坐标系并获取对应的像素值
    points_and_pixel_values = []
    for point in points_inside_polygon:
        y, x = point
        pixel_value = image[y + min_y][x + min_x]
        points_and_pixel_values.append(((x + min_x, y + min_y), pixel_value))
    return points_and_pixel_values

def count_connected_components(imgggg):
    imgggg[np.where(imgggg >= 14)] = 255
    imgggg[np.where(imgggg <= 13)] = 0
    # 检测连通区域
    num_labels, _ = cv2.connectedComponents(imgggg + 1)
    # 返回连通区域个数（减2是因为背景 & 墙也被视为连通区域）
    return num_labels - 2


# 提取所有边的函数
def extract_edges(coords, adjacency_mat):
    edges = []
    for i in range(len(adjacency_mat)):
        for j in range(i + 1, len(adjacency_mat[i])):
            if adjacency_mat[i][j]:
                edge = (coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
                edges.append(edge)
    return edges
def random_color(variance_threshold):
    color = [0, 0, 0]
    while np.var(color) < variance_threshold:
        color = [random.randint(0, 255) for _ in range(3)]
    return tuple(color)

def draw_edges(image, edges):
    for edge in edges:
        x1, y1, x2, y2 = edge
        color = random_color(variance_threshold=25)
        cv2.line(image, (x1.astype(np.int32), y1.astype(np.int32)), (x2.astype(np.int32), y2.astype(np.int32)), color, 1)
    return image

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

def get_cycle_basis_and_semantic(output_points_ori, output_edges_ori):
    output_points = [tuple(p + [99999, 99999, 99999, 99999]) for p in output_points_ori.tolist()]
    output_edges = [tuple([tuple([e[0], e[1]] + [99999, 99999, 99999, 99999]), tuple([e[2], e[3]] + [99999, 99999, 99999, 99999])]) for e in output_edges_ori]

    output_points = copy.deepcopy(output_points)
    output_edges = copy.deepcopy(output_edges)
    # print(output_points)
    # print(output_edges)
    # assert 0
    # 一个关于索引和输出点的字典
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index # 这里无法处理重复点，不能去掉nms
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point # 这里无法处理重复点，不能去掉nms
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))
    # print(d)

    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])

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
    # print(connected_components)
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
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c] # 固定的边集，不会删除
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c) # 用于遍历以减少时间复杂度的边集，其中的边会删除
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
                            if edge[0] == current_point or edge[1] == current_point:
                                relevant_edges.append(edge)
                        # 求当前点的所有邻边对应的出射角度
                        relevant_edges_degree = []
                        for relevant_edge in relevant_edges:
                            # 出射向量
                            vec = None
                            if relevant_edge[0] == current_point:
                                vec = (relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                            elif relevant_edge[1] == current_point:
                                vec = (relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
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
                                vec_from_current_point_to_last_point = (relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                relevant_edges.remove(relevant_edge)
                                relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                            elif relevant_edge == (last_point, current_point):
                                vec_from_current_point_to_last_point = (relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                relevant_edges.remove(relevant_edge)
                                relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                            else:
                                continue
                        # 求从这个角度开始逆时针旋转，碰到的最后一个当前点的邻边
                        # 同时把没扫过部分（就是内角）的角度区域记录下来
                        # 并根据角度区域和预测语义标签，求出当前角对应的内角语义
                        rotate_deltas_counterclockwise = []
                        # 记录内角区域，逆时针，从前一个角度到后一个角度
                        interior_angles = []
                        for relevant_edge_degree in relevant_edges_degree:
                            rotate_delta = rotate_degree_counterclockwise_from_counter_degree(vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                            rotate_deltas_counterclockwise.append(rotate_delta)
                            interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                        # print(rotate_deltas_counterclockwise)
                        # 最大角对应索引
                        max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                        # 找到对应的内角
                        interior_angle_counterclockwise = interior_angles[max_rotate_index]
                        # 求出对应的语义区域
                        # 先求出当前点的所有语义，顺序按照四个象限排序
                        current_point_semantic = [current_point[3], current_point[2], current_point[5], current_point[4]]
                        # 求出该逆时针角占了四个象限的多少角度
                        # 求法：直接求度数较小的逆时针转到度数较大的覆盖四象限角度
                        # 然后判断，如果度数较小的是内角区域的“源方向”，则正好是覆盖四象限角度；
                        # 如果度数较小的是内角区域的“目标方向”，则对覆盖四象限角度用90度减去；
                        interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise) # 度数较小的
                        interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # 度数较大的
                        quadrant_smaller_to_bigger_counterclockwise = get_quadrant((interior_angle_counterclockwise_degree_smaller,
                                                                                    interior_angle_counterclockwise_degree_bigger))
                        # print(quadrant_smaller_to_bigger_counterclockwise)
                        if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                            pass
                        elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                            quadrant_smaller_to_bigger_counterclockwise = (90 - quadrant_smaller_to_bigger_counterclockwise[0],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[1],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[2],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[3])
                        else:
                            assert 0
                        # 判断如果小于45度，则将语义置为-1
                        current_point_semantic_valid = []
                        for qd, seman in enumerate(current_point_semantic):
                            if quadrant_smaller_to_bigger_counterclockwise[qd] >= 45:
                                current_point_semantic_valid.append(seman)
                            else:
                                current_point_semantic_valid.append(-1)
                        # 对有效内角语义进行统计
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
                                    output_edges_c_copy_for_traversing.remove((d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                elif (d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                    output_edges_c_copy_for_traversing.remove((d_rev[edge_number[1]], d_rev[edge_number[0]]))
                    # 算面积时不需要闭环
                    simple_cycle.pop(-1)
                    simple_cycle_number.pop(-1)
                    # 存起来（逆时针计算面积，如果面积为负则不加入，说明是最大的那个）
                    polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                    polygon_counterclockwise.pop(-1)
                    # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                    if poly_area(polygon_counterclockwise) > 0:
                    # if 1:
                        simple_cycles_c.append(simple_cycle)
                        simple_cycles_number_c.append(simple_cycle_number)
                        # 民主投票统计语义（最大的那个圈就不用算了），得到该simple_cycle的语义并记录
                        semantic_result = {}
                        for semantic_label in range(99998, 100000):
                            semantic_result[semantic_label] = 0
                        for everypoint_semantic in simple_cycle_semantics:
                            everypoint_semantic = [s for s in everypoint_semantic if s != -1]
                            for label in everypoint_semantic:
                                semantic_result[label] += 1 / len(everypoint_semantic)
                        # print(semantic_result)
                        # 如果最高票相同则等概率随机选一个（11和12不算）
                        this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                        # print(this_cycle_semantic)
                        this_cycle_result = None
                        if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                            # 以唯一最高票为准
                            this_cycle_result = this_cycle_semantic[0][0]
                        else:
                            # 找出所有最高票数并等概率随机抽一个
                            this_cycle_results = [i[0] for i in this_cycle_semantic if i[1] == this_cycle_semantic[0][1]]
                            this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                        # print(this_cycle_result)
                        simple_cycle_semantics_c.append(this_cycle_result)
                    if poly_area(polygon_counterclockwise) < 0: # 最大的那个圈
                        # if 1:
                        simple_cycles_c.append(simple_cycle + ['max'])
                        simple_cycles_number_c.append(simple_cycle_number)
                        # 民主投票统计语义（就不用算了），得到该simple_cycle的语义并记录
                        semantic_result = {}
                        for semantic_label in range(99998, 100000):
                            semantic_result[semantic_label] = 0
                        for everypoint_semantic in simple_cycle_semantics:
                            everypoint_semantic = [s for s in everypoint_semantic if s != -1]
                            for label in everypoint_semantic:
                                semantic_result[label] += 1 / len(everypoint_semantic)
                        # print(semantic_result)
                        # 如果最高票相同则等概率随机选一个（11和12不算）
                        this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                        # print(this_cycle_semantic)
                        this_cycle_result = None
                        if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                            # 以唯一最高票为准
                            this_cycle_result = this_cycle_semantic[0][0]
                        else:
                            # 找出所有最高票数并等概率随机抽一个
                            this_cycle_results = [i[0] for i in this_cycle_semantic if i[1] == this_cycle_semantic[0][1]]
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

def get_cycle_basis_and_semantic_3_semansimplified(output_points_ori, output_edges_ori):
    output_points = [tuple(p + [99999, 99999, 99999, 99999]) for p in output_points_ori.tolist()]
    output_edges = [
        tuple([tuple([e[0], e[1]] + [99999, 99999, 99999, 99999]), tuple([e[2], e[3]] + [99999, 99999, 99999, 99999])])
        for e in output_edges_ori]

    output_points = copy.deepcopy(output_points)
    output_edges = copy.deepcopy(output_edges)
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

for sample_id in tqdm(range(0, 80788)):
    '''read image'''
    image = cv2.imread(os.path.join(image_path, str(sample_id) + '.png'), cv2.IMREAD_UNCHANGED) # 4 channels

    '''随机提取四通道像素
        Pixel 1: [ 0 13  0  0]
        Pixel 2: [  3   0   0 255]
        Pixel 3: [  3   0   0 255]
        Pixel 4: [  3   0   0 255]
        Pixel 5: [ 0 13  0  0]
        Pixel 6: [  4   7   0 255]
        Pixel 7: [ 0 13  0  0]
        Pixel 8: [ 0 13  0  0]
        Pixel 9: [  3   0   0 255]
        Pixel 10: [ 0 13  0  0]
        ...
    label.xlsx  channels order: 3 2 1 4
    '''



    '''read corners annotation'''
    if sample_id in train:
        graph = np.load('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-withsemantics/train/' + str(sample_id) + '.npy', allow_pickle=True).item()

    elif sample_id in test:
        graph = np.load('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-withsemantics/test/' + str(sample_id) + '.npy', allow_pickle=True).item()
    else:
        continue

    print(np.load('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-edgesemantics/train/' + str(2) + '.npy', allow_pickle=True).item())
    assert 0

    corners = graph['corners_np']
    adjacency_matrix = graph['adjacency_matrix_np']
    assert np.all(adjacency_matrix == adjacency_matrix.T)
    # print(corners)
    # print(adjacency_matrix)
    # 提取所有边
    edges_list = extract_edges(corners, adjacency_matrix)
    assert len(edges_list) == np.sum(np.triu(adjacency_matrix))
    #
    # # 打印边
    # print("Edges:")
    # for edge in edges_list:
    #     x1, y1, x2, y2 = edge
    #     print(f"({x1}, {y1}), ({x2}, {y2})")

    # # 先尝试把这堆边可视化（在三通道bin_imgs上随机色彩）
    channel_2 = copy.deepcopy(image[:, :, 1]) # 提取第二通道（索引为1）
    # thresholded_channel_2 = np.where(channel_2 >= 14, 255, 0) # 将像素值大于等于14的设置为255，否则设置为0
    # three_channel_img = np.stack((thresholded_channel_2, thresholded_channel_2, thresholded_channel_2), axis=-1).astype(np.uint8) # 复制thresholded_channel_2成三通道图像
    #
    # # 在三通道图像上绘制边缘
    # colored_edges_img = draw_edges(three_channel_img, edges_list)
    # cv2.imwrite(r'/home/myubt/Projects/house_diffusion-main/Housegan-data-reader-main/edges_visualize/' + str(sample_id) + '.png', colored_edges_img)

    d_rev, simple_cycles_, simple_cycles_semantics = get_cycle_basis_and_semantic(corners, edges_list)
    # d_rev, simple_cycles_, simple_cycles_semantics = get_cycle_basis_and_semantic_3_semansimplified(corners, edges_list)

    simple_cycles = []
    for sc_ in simple_cycles_:
        if sc_[-1] != 'max':
            simple_cycles.append([[int(_) for _ in list(t)[:2]] for t in sc_])
        else:
            simple_cycles.append([[int(_) for _ in list(t)[:2]] for t in sc_[:-1]] + ['max'])
    # for sc in simple_cycles:
    #     print(sc)

    ''' 需要检测多边形提取的数量对不对！
方案：检测bin_imgs中的连通域数量，和提取的多边形数量作对比，如果数量不对，可能需要忍痛删除部分数据。'''
    # 读取bin_imgs图像，检测连通域
    true_polygon_number = count_connected_components(channel_2)
    # 多边形数量
    r2g_obtain_polygon_number = len(simple_cycles)
    # if not true_polygon_number == r2g_obtain_polygon_number:
    #     print(str(sample_id)) # 已判明r2g的多边形提取是完全正确的

    ''' 下一步就是找出每个多边形的语义是什么 
    注意最大圈的标签直接是13'''
    polygon_semantics = []
    for polygon in simple_cycles:
        if polygon[-1] != 'max':
            points_and_pixel_values = get_points_and_pixel_values_inside_polygon(copy.deepcopy(image[:, :, 1]), polygon)
            lbl = get_label(points_and_pixel_values)
            polygon_semantics.append((polygon[:-1], lbl))
        else:
            polygon_semantics.append((polygon[:-2], 13))
    # print(polygon_semantics)
    # assert 0
    '''角点字典，给每个角点一个14维(0-13)向量，初始值都是0,房间计数 '''
    seman_d = {}
    for corner in corners:
        corner_tupleint = tuple(corner.astype(np.int32).tolist())
        # print(corner, corner_tupleint)
        seman_d[corner_tupleint] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for polygon_semantic in polygon_semantics:
        polygon = polygon_semantic[0]
        semantic = polygon_semantic[1]
        # print(polygon)
        # print(semantic)
        for point in polygon:
            seman_d[tuple(point)][semantic] += 1
    # for k, v in seman_d.items():
    #     print(k, v)

    ''' write'''
    new_graph = copy.deepcopy(graph)
    new_graph['semantics'] = seman_d
    if sample_id in train:
        np.save('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-edgesemantics/train/' + str(sample_id) + '.npy', new_graph)
    elif sample_id in test:
        np.save('/home/myubt/Projects/house_diffusion-main/datasets/rplang-v2-edgesemantics/test/' + str(sample_id) + '.npy', new_graph)