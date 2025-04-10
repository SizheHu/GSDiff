import copy
import os
import cv2
import numpy as np
import random
import shutil
from tqdm import *
np.set_printoptions(threshold=np.inf)

assert os.path.exists('rplandata'), 'path not exist'
assert os.path.exists('rplandata/Data'), 'path not exist'

# At this point you need to confirm that the floorplan_dataset folder (containing 80,788 images) is in the rplandata/Data directory
assert os.path.exists('rplandata/Data/floorplan_dataset'), 'path not exist'

# all data
all_data = os.listdir(r'rplandata/Data/floorplan_dataset')

# Iterate over each file name fn in all_data, remove the last five characters of the file name and convert it to an integer.
# If the file name is not 'list.txt', add it to the ids list
ids = [int(fn[:-4]) for fn in all_data]
# print(ids)

# Create a path to place the images
if not os.path.exists('rplandata/Data/3-channel-semantics-256'):
    os.mkdir('rplandata/Data/3-channel-semantics-256')
if not os.path.exists('rplandata/Data/1-channel-semantics-256'):
    os.mkdir('rplandata/Data/1-channel-semantics-256')
if not os.path.exists('rplandata/Data/bin_imgs'):
    os.mkdir('rplandata/Data/bin_imgs')
if not os.path.exists('rplandata/Data/e_imgs'):
    os.mkdir('rplandata/Data/e_imgs')
if not os.path.exists('rplandata/Data/e_imgs_filteredv1'):
    os.mkdir('rplandata/Data/e_imgs_filteredv1')
if not os.path.exists('rplandata/Data/e_imgs_filteredv2'):
    os.mkdir('rplandata/Data/e_imgs_filteredv2')
if not os.path.exists('rplandata/Data/e_imgs_filteredv3'):
    os.mkdir('rplandata/Data/e_imgs_filteredv3')

# Iterate through the list of ids
for i, id in tqdm(enumerate(ids)):
    # Use OpenCV to read the file. The file path is rplandata/Data/floorplan_dataset/ plus the string form of the id and the file extension '.png'
    origin_img = cv2.imread('rplandata/Data/floorplan_dataset/' + str(id) + '.png', -1)[:, :, 1]


    # Assign the processed image to the semantics variable
    semantics = origin_img
    # Save the semantics image to the path 'rplandata/Data/3-channel-semantics-256/', and the file name is id plus '.png'
    cv2.imwrite('rplandata/Data/3-channel-semantics-256/' + str(id) + '.png', semantics)
    # Get the first channel of the semantics image (index 0), which is usually a grayscale image.
    semantics2 = origin_img
    # Save the single-channel image to the path 'rplandata/Data/1-channel-semantics-256/', and the file name is id plus '.png'
    cv2.imwrite('rplandata/Data/1-channel-semantics-256/' + str(id) + '.png', semantics2)

    # # Set all pixels with a value greater than or equal to 14 in the first channel to red
    origin_img[np.where(origin_img[:, :] >= 14)] = 255
    # # Set all pixels with a value less than or equal to 13 in the first channel to black
    origin_img[np.where(origin_img[:, :] <= 13)] = 0
    # Extract the first channel of the processed image
    bin_img = origin_img
    # Save the channel to the 'bin_imgs/' path
    cv2.imwrite('rplandata/Data/bin_imgs/' + str(id) + '.png', bin_img)

    # Assign the binary image to the dst variable
    dst = bin_img
    # Create a 3x3 all-1 convolution kernel
    kernel_erode = np.ones((3, 3))
    # Use the corrosion operation to process the dst image and assign the result to dst again
    dst = cv2.erode(dst, kernel_erode, iterations=1)



def fix_cv2_bug(bin_img):
    # Fix a bug in cv2 where it returns a new image by creating a deep copy of bin_img and incrementing it by 1
    return copy.deepcopy(bin_img) + 1

def isvalid(img):
    kernel = np.ones((2, 2), dtype=img.dtype) * 255
    # Iterate over each 2x2 region in the image, if it is the same as the core, the image is invalid
    for i in range(0, 255):
        for j in range(0, 255):
            if (img[i:i + 2, j:j + 2] == kernel).all():
                return False
    # If no region is found that is identical to the core, the image is valid
    return True

def isvalid2(img):
    # Traverse every pixel in the image, except for the edge
    for i in range(1, 255):
        for j in range(1, 255):
            # If the current pixel value is 255 (white)
            if img[i, j] == 255:
                # Check if the sum of four adjacent pixels is 255 or 0, if so, the image is invalid
                if img[i - 1, j] + img[i , j - 1] + img[i + 1, j] + img[i, j + 1] == 255 or \
                        img[i - 1, j] + img[i , j - 1] + img[i + 1, j] + img[i, j + 1] == 0:
                    return False
    # If no invalid condition is detected, the image is valid
    return True







# 获取'bin_imgs'目录下所有文件
bin_imgs = os.listdir('rplandata/Data/bin_imgs')
# 初始化计数器
count = 0
# 遍历文件
for fn in tqdm(bin_imgs):
    # 文件计数
    count += 1
    # 初始化最终图像
    final = None
    # 以灰度模式读取二值图像
    bin_img = cv2.imread('rplandata/Data/bin_imgs/' + fn, cv2.IMREAD_GRAYSCALE)
    # 打印出图像计数、文件名、图像尺寸及一个随机像素点的值，用于测试数据正确性
    # print(count, fn, bin_img.shape, bin_img[random.randint(0, 255), random.randint(0, 255)])
    # 将读取的图像写入文件以保存

    # cv2.imwrite('./rplandata/Data/t0_' + fn, bin_img)
    # 初始化生命周期变量
    life = 31
    # 当生命周期不为0时执行循环
    while life >= 0:
        # 生命周期递减
        life -= 1
        # 获取连通区域的数量
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(fix_cv2_bug(bin_img), connectivity=8)
        # print('num', num, stats)
        # 试图通过腐蚀操作简化图像
        kernel_erode = np.ones((3, 3)) # 定义腐蚀操作的核
        eroded = cv2.erode(bin_img, kernel_erode, iterations=1) # 对图像进行腐蚀操作

        # 获取腐蚀后的图像的连通区域数量
        num_e, labels_e, stats_e, centroids_e = cv2.connectedComponentsWithStats(fix_cv2_bug(eroded), connectivity=8)
        # 如果腐蚀后图像变成全黑，则认为操作完成
        if np.sum(eroded) == 0:
            # 将未腐蚀的图像设为最终图像
            final = bin_img
            # 尝试移除孤立的白色像素以平滑黑色区域
            unflattened_patterns_4 = [[[0, 255, 0], [0, 0, 0], [0, 0, 0]],
                                      [[0, 0, 0], [255, 0, 0], [0, 0, 0]],
                                      [[0, 0, 0], [0, 0, 0], [0, 255, 0]],
                                      [[0, 0, 0], [0, 0, 255], [0, 0, 0]],
                                      ]
            for i in range(1, 255):
                for j in range(1, 255):
                    # 如果检测到上述模式，则将该部分平滑为黑色
                    if final[i - 1: i + 2, j - 1: j + 2].tolist() in unflattened_patterns_4:
                        final[i - 1: i + 2, j - 1: j + 2] = 0
            # 将最终图像写入文件保存

            cv2.imwrite('./rplandata/Data/e_imgs/t999_' + fn, final)
            break # 退出生命周期
        else:
            if num_e < num:
                # 如果腐蚀后连通区域减少，则忽略腐蚀操作
                # 使用3x3滑动窗口核识别宽度至少为3个白色像素的区域
                thick_white = np.zeros((256, 256), dtype=bin_img.dtype)
                for i in range(1, 255):
                    for j in range(1, 255):
                        if (bin_img[i - 1: i + 2, j - 1: j + 2] == 255).all():
                            thick_white[i - 1: i + 2, j - 1: j + 2] = 1

                # 复制原始图像并将宽白色区域添加到其中，从而得到1像素宽的白色线条
                bin_img_copy = copy.deepcopy(bin_img)
                thin_white = bin_img_copy + thick_white

                # 对细白色线条进行膨胀操作并添加回原图，由于可能会出现值为254的像素，因此需要重新二值化
                kernel_dilate = np.ones((3, 3))
                dilated = cv2.dilate(thin_white, kernel_dilate, iterations=1)
                bin_img += dilated
                ret, bin_img = cv2.threshold(bin_img, thresh=128, maxval=255, type=cv2.THRESH_BINARY)

                # 平滑轮廓
                unflattened_patterns_3 = [[[255, 255, 255], [255, 0, 255], [0, 0, 0]],
                                        [[0, 0, 0], [255, 0, 255], [255, 255, 255]],
                                        [[255, 255, 0], [255, 0, 0], [255, 255, 0]],
                                        [[0, 255, 255], [0, 0, 255], [0, 255, 255]],

                                          [[255, 255, 255], [255, 0, 0], [0, 0, 0]],
                                          [[255, 255, 255], [0, 0, 255], [0, 0, 0]],
                                          [[0, 0, 0], [0, 0, 255], [255, 255, 255]],
                                          [[0, 0, 0], [255, 0, 0], [255, 255, 255]],
                                          [[255, 255, 0], [255, 0, 0], [255, 0, 0]],
                                          [[255, 0, 0], [255, 0, 0], [255, 255, 0]],
                                          [[0, 255, 255], [0, 0, 255], [0, 0, 255]],
                                          [[0, 0, 255], [0, 0, 255], [0, 255, 255]],

                                          [[255, 255, 0], [255, 0, 0], [0, 0, 0]],
                                          [[0, 255, 255], [0, 0, 255], [0, 0, 0]],
                                          [[0, 0, 0], [0, 0, 255], [0, 255, 255]],
                                          [[0, 0, 0], [255, 0, 0], [255, 255, 0]],
                                          ]
                for i in range(1, 255):
                    for j in range(1, 255):
                        if bin_img[i - 1: i + 2, j - 1: j + 2].tolist() in unflattened_patterns_3:
                            bin_img[i, j] = 255
                # 平滑轮廓
                unflattened_patterns_5 = [[[255, 0, 255], [255, 255, 255], [255, 255, 255]],
                                          [[255, 255, 255], [0, 255, 255], [255, 255, 255]],
                                          [[255, 255, 255], [255, 255, 0], [255, 255, 255]],
                                          [[255, 255, 255], [255, 255, 255], [255, 0, 255]],
                                          ]
                for i in range(1, 255):
                    for j in range(1, 255):
                        if bin_img[i - 1: i + 2, j - 1: j + 2].tolist() in unflattened_patterns_5:
                            bin_img[i - 1: i + 2, j - 1: j + 2] = 255

            else:
                # 如果腐蚀后连通区域没有减少，使用腐蚀后的图像继续下一轮循环
                bin_img = copy.deepcopy(eroded)



# 使用2*2白色核心过滤错误数据（从75350降至71814）

for fn in tqdm(os.listdir('rplandata/Data/e_imgs')):
    # 将筛选后的图片复制到新目录
    shutil.copy('rplandata/Data/e_imgs/' + fn, 'rplandata/Data/e_imgs_filteredv1/' + fn.replace('t999_', ''))
count = 0
remove1_count = 0
for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv1')):
    count += 1
    # 打印当前处理的文件数和文件名
    # print(count, fn)
    img = cv2.imread('rplandata/Data/e_imgs_filteredv1/' + fn, cv2.IMREAD_GRAYSCALE)
    if not isvalid(img):
        # 如果图片不合法（指图片中有2*2白色块，说明腐蚀不彻底）则删除，并计数
        os.remove('rplandata/Data/e_imgs_filteredv1/' + fn)
        remove1_count += 1
        # 打印已删除的文件数量
        # print(remove1_count)


# 过滤拓扑错误（死胡同）（从71814降至71763）

for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv1')):
    shutil.copy('rplandata/Data/e_imgs_filteredv1/' + fn, 'rplandata/Data/e_imgs_filteredv2/' + fn)
count = 0
remove2_count = 0
for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv2')):
    count += 1
    # print(count, fn)
    img = cv2.imread('rplandata/Data/e_imgs_filteredv2/' + fn, cv2.IMREAD_GRAYSCALE)
    if not isvalid2(img):
        os.remove('rplandata/Data/e_imgs_filteredv2/' + fn)
        remove2_count += 1
        # print(remove2_count)



# 确保与原始拓扑结构一致（8-连通数）（因此可以自然获得语义和门的信息）（71763维持不变）

for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv2')):
    shutil.copy('rplandata/Data/e_imgs_filteredv2/' + fn, 'rplandata/Data/e_imgs_filteredv3/' + fn)
count = 0
remove3_count = 0
for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv3')):
    count += 1
    # print(count, fn)
    img = cv2.imread('rplandata/Data/e_imgs_filteredv3/' + fn, cv2.IMREAD_GRAYSCALE)
    img_ori = cv2.imread('rplandata/Data/bin_imgs/' + fn, cv2.IMREAD_GRAYSCALE)
    if not (cv2.connectedComponentsWithStats(fix_cv2_bug(img), connectivity=8)[0] == cv2.connectedComponentsWithStats(fix_cv2_bug(img_ori), connectivity=8)[0]):
        os.remove('rplandata/Data/e_imgs_filteredv3/' + fn)
        remove3_count += 1
        # print(remove3_count)






# 提取结构图、内部门、边界、前门、房间语义等信息
# 定义结构图字典，键为文件名，值为图结构（图结构以字典形式表示，键为(x1, y1)，值为[上，左，下，右]方向的相邻结点坐标(xi, yi)或(-1, -1)表示无邻接结点）
count = 0
structure_graphs = {}

# 获取两个角点之间的坐标序列
def get_coords(corner1, corner2):
    # 如果两个角点在同一列
    if corner1[0] == corner2[0]:
        if corner1[1] < corner2[1]:
            return [(corner1[0], i) for i in range(corner1[1], corner2[1] + 1)]
        elif corner1[1] > corner2[1]:
            return [(corner1[0], i) for i in range(corner2[1], corner1[1] + 1)]
        else:
            assert 0
    # 如果两个角点在同一行
    elif corner1[1] == corner2[1]:
        if corner1[0] < corner2[0]:
            return [(j, corner1[1]) for j in range(corner1[0], corner2[0] + 1)]
        elif corner1[0] > corner2[0]:
            return [(j, corner1[1]) for j in range(corner2[0], corner1[0] + 1)]
        else:
            assert 0
    else:
        assert 0

# 判断坐标序列是否为边界的函数
def is_edge_func2(coords, corners, img):
    for coord in coords:
        if img[coord[1], coord[0]] == 0 or (coord in corners and coord != coords[0] and coord != coords[-1]):
            return False
    return True


# 遍历已过滤的图像，提取结构图
for fn in tqdm(os.listdir('rplandata/Data/e_imgs_filteredv3')):
    count += 1
    # print(count, fn)
    img = cv2.imread('rplandata/Data/e_imgs_filteredv3/' + fn, cv2.IMREAD_GRAYSCALE)
    try:
        # 初始化角点列表
        corners_L = []
        corners_T = []
        corners_X = []
        # 提取角点
        for i in range(1, 255):
            for j in range(1, 255):
                # 没有I形交点，只考虑L、T、X形交点
                if img[i, j] == 255:
                    # L
                    if img[i - 1, j] + img[i , j - 1] + img[i + 1, j] + img[i, j + 1] == 254 and \
                            img[i - 1, j] + img[i + 1, j] == 255:
                        corners_L.append((j, i))
                    # T
                    elif img[i - 1, j] + img[i , j - 1] + img[i + 1, j] + img[i, j + 1] == 253:
                        corners_T.append((j, i))
                    # X
                    elif img[i - 1, j] + img[i , j - 1] + img[i + 1, j] + img[i, j + 1] == 252:
                        corners_X.append((j, i))
                    else:
                        continue
                else:
                    continue
        # 合并所有角点列表
        corners = []
        corners.extend(corners_L)
        corners.extend(corners_T)
        corners.extend(corners_X)
        # 提取边界
        edges = []
        for corner1 in corners:
            for corner2 in corners:
                if corner1 != corner2:
                    # 如果两个角点不在同一行或同一列，则跳过
                    if not ((corner1[0] == corner2[0]) or (corner1[1] == corner2[1])):
                        continue
                    else:
                        # 获取两个角点间的坐标序列
                        coords = get_coords(corner1, corner2)
                        # 如果不是最小墙段
                        if not is_edge_func2(coords, corners, img):
                            continue
                        else:
                            # 将坐标序列添加到边界列表
                            edges.append((corner1, corner2))
                else:
                    continue
        # 将边界转换为结构图
        structure_graph = {}
        for corner in corners:
            # 获取相邻点
            adjacents = {}
            for edge in edges:
                # 判断方向
                if edge[0] == corner or edge[1] == corner:
                    e_l = list(edge)
                    e_l.remove(corner)
                    adjacent = e_l[0]
                    # up
                    if adjacent[0] == corner[0] and adjacent[1] < corner[1]:
                        adjacents['up'] = adjacent
                    # down
                    elif adjacent[0] == corner[0] and adjacent[1] > corner[1]:
                        adjacents['down'] = adjacent
                    # left
                    elif adjacent[1] == corner[1] and adjacent[0] < corner[0]:
                        adjacents['left'] = adjacent
                    # right
                    elif adjacent[1] == corner[1] and adjacent[0] > corner[0]:
                        adjacents['right'] = adjacent
                    else:
                        assert 0
            # 将相邻点信息添加到结构图
            adjacents_list = []
            for direction in ['up', 'left', 'down', 'right']:
                if direction in adjacents.keys():
                    adjacents_list.append(adjacents[direction])
                else:
                    adjacents_list.append((-1, -1))
            structure_graph[corner] = adjacents_list
        # 将结构图添加到字典中
        structure_graphs[int(fn[:-4])] = structure_graph
    except:
        pass

# 将结构图字典保存为.npy文件
np.save('rplandata/Data/structure_graphs.npy', structure_graphs)


# 加载看看
b = np.load('rplandata/Data/structure_graphs.npy', allow_pickle=True).item()
print(b)