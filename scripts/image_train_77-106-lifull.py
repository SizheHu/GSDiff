import sys
sys.path.append('/home/user00/HSZ/gsdiff-main-lifull')
sys.path.append('/home/user00/HSZ/gsdiff-main-lifull/datasets')
sys.path.append('/home/user00/HSZ/gsdiff-main-lifull/gsdiff')
sys.path.append('/home/user00/HSZ/gsdiff-main-lifull/scripts/metrics')


import math
import torch
import shutil
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from itertools import cycle
from datasets.lifull_55_100 import *
from datasets.lifull import lifull
from gsdiff.heterhouse_75_106_lifull import HeterHouseModel
from gsdiff.heterhouse_56_11_lifull import EdgeModel
from gsdiff.utils import *
import torch.nn.functional as F
from scripts.metrics.fid import fid
from scripts.metrics.kid2 import kid2


diffusion_steps = 1000
lr = 1e-4
weight_decay = 0 # 以后还是把它加上
total_steps = 1000000
batch_size = 256 # 256
batch_size_val = 3000 
device = 'cuda:1' # 服务器
merge_points = True
clamp_trick_training = True


def map_to_binary(tensor):
    batch_size, n_values = tensor.shape
    binary_tensor = torch.zeros((batch_size, n_values, 12), dtype=torch.float32, device=tensor.device)

    # 创建一个mask，标记出非99999的值
    mask = tensor != 99999

    # 处理非99999的值
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))

    # 分离整数和小数部分
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part

    # 处理整数部分
    for i in range(8):
        binary_tensor[:, :, 7 - i] = integer_part % 2
        integer_part //= 2

    # 处理小数部分
    fractional_part *= 16  # 将小数部分扩展到4位二进制数
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(4):
        binary_tensor[:, :, 11 - i] = fractional_part % 2
        fractional_part //= 2

    # 使用mask确保原始为99999的值在二值向量中为0
    binary_tensor = torch.where(mask.unsqueeze(-1), binary_tensor, torch.zeros_like(binary_tensor))

    return binary_tensor

def map_to_fournary(tensor):
    batch_size, n_values = tensor.shape
    fournary_tensor = torch.zeros((batch_size, n_values, 6), dtype=torch.float32, device=tensor.device)

    # 创建一个mask，标记出非99999的值
    mask = tensor != 99999

    # 处理非99999的值
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))

    # 分离整数和小数部分
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part

    # 处理整数部分
    for i in range(4):
        fournary_tensor[:, :, 3 - i] = integer_part % 4
        integer_part //= 4

    # 处理小数部分
    fractional_part *= 16  # 将小数部分扩展到2位4进制数
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(2):
        fournary_tensor[:, :, 5 - i] = fractional_part % 4
        fractional_part //= 4

    # 使用mask确保原始为99999的值在二值向量中为0
    fournary_tensor = torch.where(mask.unsqueeze(-1), fournary_tensor, torch.zeros_like(fournary_tensor))

    return fournary_tensor

def map_to_eightnary(tensor):
    batch_size, n_values = tensor.shape
    eightnary_tensor = torch.zeros((batch_size, n_values, 5), dtype=torch.float32, device=tensor.device)

    # 创建一个mask，标记出非99999的值
    mask = tensor != 99999

    # 处理非99999的值
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))

    # 分离整数和小数部分
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part

    # 处理整数部分
    for i in range(3):
        eightnary_tensor[:, :, 2 - i] = integer_part % 8
        integer_part //= 8

    # 处理小数部分
    fractional_part *= 64  # 将小数部分扩展到2位8进制数
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(2):
        eightnary_tensor[:, :, 4 - i] = fractional_part % 8
        fractional_part //= 8

    # 使用mask确保原始为99999的值在二值向量中为0
    eightnary_tensor = torch.where(mask.unsqueeze(-1), eightnary_tensor, torch.zeros_like(eightnary_tensor))

    return eightnary_tensor

def map_to_sxtnary(tensor):
    batch_size, n_values = tensor.shape
    sxtnary_tensor = torch.zeros((batch_size, n_values, 3), dtype=torch.float32, device=tensor.device)

    # 创建一个mask，标记出非99999的值
    mask = tensor != 99999

    # 处理非99999的值
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))

    # 分离整数和小数部分
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part

    # 处理整数部分
    for i in range(2):
        sxtnary_tensor[:, :, 1 - i] = integer_part % 16
        integer_part //= 16

    # 处理小数部分
    fractional_part *= 16  # 将小数部分扩展到1位16进制数
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(1):
        sxtnary_tensor[:, :, 2 - i] = fractional_part % 16
        fractional_part //= 16

    # 使用mask确保原始为99999的值在二值向量中为0
    sxtnary_tensor = torch.where(mask.unsqueeze(-1), sxtnary_tensor, torch.zeros_like(sxtnary_tensor))

    return sxtnary_tensor


'''create output_dir'''
output_dir = 'outputs/structure-77-106-lifull/'
os.makedirs(output_dir, exist_ok=False)

'''Diffusion Settings'''
# cosine beta
alpha_bar = lambda t: math.cos((t) / 1.000 * math.pi / 2) ** 2
betas = []
max_beta = 0.999
for i in range(diffusion_steps):
    t1 = i / diffusion_steps
    t2 = (i + 1) / diffusion_steps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
betas = np.array(betas, dtype=np.float64)
alphas = 1.0 - betas  # [a1, a2, ..., a1000]: 1- -> 0+

# gaussian diffuse settings
alphas_cumprod = np.cumprod(alphas)  # [a1, a1a2, a1a2a3, ..., a1a2...a1000] 1- -> 0+
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])  # [1, a1, a1a2, a1a2a3, ..., a1a2...a999]
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)  # sqrt([a1, a1a2, a1a2a3, ..., a1a2...a1000]) 1- -> 0+
sqrt_one_minus_alphas_cumprod = np.sqrt(
    1.0 - alphas_cumprod)  # sqrt([1-a1, 1-a1a2, 1-a1a2a3, ..., 1-a1a2...a1000]) 0+ -> 1-
# print(1.0 - alphas_cumprod) # 0+ -> 1-
sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)  # sqrt([1/a1, 1/a1a2, 1/a1a2a3, ..., 1/a1a2...a1000])
sqrt_recipm1_alphas_cumprod = np.sqrt(
    1.0 / alphas_cumprod - 1)  # sqrt([1/a1 - 1, 1/a1a2 - 1, 1/a1a2a3 - 1, ..., 1/a1a2...a1000 - 1])
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod)  # variance for posterior q(x_{t-1} | x_t, x_0)
posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)  # posterior distribution mean
posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (
            1.0 - alphas_cumprod)  # posterior distribution mean

'''Data'''
dataset_train = Lifull_55_100('train')
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=64,
                        drop_last=True, pin_memory=True)  # try different num_workers to be faster
dataloader_train_iter = iter(cycle(dataloader_train))
dataset_val = Lifull_55_100('val')
dataloader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=True)  # try different num_workers to be faster
dataloader_val_iter = iter(cycle(dataloader_val))

dataset_val_for_gt_rendering = lifull('val')
dataloader_val_for_gt_rendering = DataLoader(dataset_val_for_gt_rendering, batch_size=batch_size_val, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=False)  # try different num_workers to be faster
dataloader_val_iter_for_gt_rendering = iter(cycle(dataloader_val_for_gt_rendering))

'''为了在验证集上计算FID、KID等指标，需要先把验证集渲染。'''
gt_dir_val = output_dir + 'val_gt' + '/'
if os.path.exists(gt_dir_val):
    shutil.rmtree(gt_dir_val)  # 删除路径
os.makedirs(gt_dir_val)  # 创建路径
# 验证集的3000个样本逐一渲染成图片
# colors = {6: (0, 0, 0), 0: (222, 241, 244), 1: (159, 182, 234), 2: (92, 112, 107), 3: (95, 122, 224),
#               4: (123, 121, 95), 5: (143, 204, 242)}
colors = {1: (0, 0, 220),
          2: (0, 220, 220),
          3: (0, 220, 0),
          4: (220, 220, 0),
          5: (220, 0, 0),
          6: (220, 0, 220),
          7: (110, 0, 110),
          8: (110, 0, 0),
          9: (0, 0, 110),
          10: (0, 110, 110),
          11: (0, 0, 0),
          12: (110, 110, 110),
          0: (20, 20, 20)}
if len(dataset_val_for_gt_rendering) % batch_size_val == 0:
    batch_numbers = len(dataset_val_for_gt_rendering) // batch_size_val
else:
    batch_numbers = len(dataset_val_for_gt_rendering) // batch_size_val + 1
for batch_count in tqdm(range(batch_numbers)):
    corners_withsemantics_0_val_batch, global_attn_matrix_val_batch, corners_padding_mask_val_batch, edges_val_batch = next(dataloader_val_iter_for_gt_rendering)
    for i in range(corners_withsemantics_0_val_batch.shape[0]):

        val_count = batch_count * batch_size_val + i
        corners_withsemantics_0_val = corners_withsemantics_0_val_batch[i][None, :, :]
        global_attn_matrix_val = global_attn_matrix_val_batch[i][None, :, :]
        corners_padding_mask_val = corners_padding_mask_val_batch[i][None, :, :]
        edges_val = edges_val_batch[i][None, :, :]

        corners_withsemantics_0_val = corners_withsemantics_0_val.clamp(-1, 1).cpu().numpy()
        corners_0_val = (corners_withsemantics_0_val[0, :, :2] * 128 + 128).astype(int)
        semantics_0_val = corners_withsemantics_0_val[0, :, 2:].astype(int)
        global_attn_matrix_val = global_attn_matrix_val.cpu().numpy()
        corners_padding_mask_val = corners_padding_mask_val.cpu().numpy()
        edges_val = edges_val.cpu().numpy()
        corners_0_val_depadded = corners_0_val[corners_padding_mask_val.squeeze() == 1][None, :, :] # (n, 2)
        semantics_0_val_depadded = semantics_0_val[corners_padding_mask_val.squeeze() == 1][None, :, :] # (n, 7)
        edges_val_depadded = edges_val[global_attn_matrix_val.reshape(1, -1, 1)][None, :, None]
        edges_val_depadded = np.concatenate((1 - edges_val_depadded, edges_val_depadded), axis=2)

        ''' get planar cycles'''
        # 形状为 (1, n, 14) 的 ndarray，包含 0 和 1;找到每个子数组中 1 所在的索引,用 99999 替换值为 0 的原始元素
        semantics_gt_i_transform_val = semantics_0_val_depadded
        semantics_gt_i_transform_indices_val = np.indices(semantics_gt_i_transform_val.shape)[-1]
        semantics_gt_i_transform_val = np.where(semantics_gt_i_transform_val == 1,
                                                    semantics_gt_i_transform_indices_val, 99999)

        gt_i_points_val = [tuple(corner_with_seman_val) for corner_with_seman_val in
                             np.concatenate((corners_0_val_depadded, semantics_gt_i_transform_val), axis=-1).tolist()[0]]

        gt_i_edges_val = edges_to_coordinates(
            np.triu(edges_val_depadded[0, :, 1].reshape(len(gt_i_points_val), len(gt_i_points_val))).reshape(-1),
            gt_i_points_val)

        d_rev_val, simple_cycles_val, simple_cycles_semantics_val = get_cycle_basis_and_semantic_3_semansimplified_lifull(
            gt_i_points_val,
            gt_i_edges_val)

        # 创建一个256x256的全白图片
        img = np.ones((256, 256, 3), np.uint8) * 255

        # draw
        for polygon_i, polygon in enumerate(simple_cycles_val):
            cv2.fillPoly(img, [np.array([[p[0], p[1]] for p in polygon], dtype=np.int32)], color=colors[simple_cycles_semantics_val[polygon_i]])
        for polygon_i, polygon in enumerate(simple_cycles_val):
            for point_i, point in enumerate(polygon):
                if point_i < len(polygon) - 1:
                    p1 = (point[0], point[1])
                    cv2.rectangle(img, (p1[0] - 3, p1[1] - 3), (p1[0] + 3, p1[1] + 3), color=(150, 150, 150), thickness=-1)
                    p2 = (polygon[point_i + 1][0], polygon[point_i + 1][1])
                    cv2.line(img, p1, p2, color=(150, 150, 150), thickness=5) # 这个地方设成5输出的才是7个像素宽

        cv2.imwrite(os.path.join(gt_dir_val, f"val_gt_{val_count}.png"), img)

print('验证集渲染完毕！')

'''Neural Network'''
model = HeterHouseModel().to(device)
print('一阶段模型参数量：', sum(p.numel() for p in model.parameters()))
'''Optim'''
optimizer = AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)
'''Training'''
step = 0
loss_curve = []
val_metrics = []

interval = 1000000 # real config
while step < total_steps:
    '''a batch of data'''
    corners_withsemantics_0, global_attn_matrix, corners_padding_mask = next(dataloader_train_iter)
    corners_withsemantics_0 = corners_withsemantics_0.to(device).clamp(-1, 1)

    global_attn_matrix = global_attn_matrix.to(device)
    corners_padding_mask = corners_padding_mask.to(device)

    # (bs, 53, 10)
    corners_withsemantics_0 = torch.cat((corners_withsemantics_0, (1 - corners_padding_mask).type(corners_withsemantics_0.dtype)), dim=2)

    '''clear all params grad, to prepare for new computation'''
    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    '''uniform t sampler'''
    t_distribution = np.ones([diffusion_steps]) / diffusion_steps  # for every t in [0, 999], uniform distribution
    t = torch.tensor(np.random.choice(len(t_distribution), size=(batch_size,), p=t_distribution),
                     dtype=torch.long, device=device)  # get random t (bs, )

    '''corners_0: (bs, 53, 2)'''
    '''corners_noise: N(0, 1)'''
    corners_withsemantics_noise = torch.randn_like(corners_withsemantics_0)
    '''t: Uniform[0, T-1], (bs, )'''
    '''corners_t: noised corners_0 (bs, 53, 2)'''
    corners_withsemantics_t = torch.tensor(sqrt_one_minus_alphas_cumprod, device=device)[t][:, None, None].expand_as(
        corners_withsemantics_noise) * corners_withsemantics_noise + \
                torch.tensor(sqrt_alphas_cumprod, device=device)[t][:, None, None].expand_as(corners_withsemantics_0) * corners_withsemantics_0


    '''model: predicts corners_noise (or corners_0) and edges_0'''
    output_corners_withsemantics1, output_corners_withsemantics2 = model(corners_withsemantics_t, global_attn_matrix, t)
    output_corners_withsemantics = torch.cat((output_corners_withsemantics1, output_corners_withsemantics2), dim=2)

    '''target'''
    corners_withsemantics_target1 = corners_withsemantics_noise
    corners_withsemantics_target2 = corners_withsemantics_0

    '''calculate loss. '''
    # 噪声L2
    corners_loss_masked1 = (corners_withsemantics_target1 - output_corners_withsemantics) ** 2

    # x0 L2
    model_variance = torch.tensor(posterior_variance, device=device)[t][:, None, None].expand_as(corners_withsemantics_t)

    pred_xstart = (
            torch.tensor(sqrt_recip_alphas_cumprod, device=device)[t][:, None, None].expand_as(
                corners_withsemantics_t) * corners_withsemantics_t -
            torch.tensor(sqrt_recipm1_alphas_cumprod, device=device)[t][:, None, None].expand_as(
                corners_withsemantics_t) * output_corners_withsemantics
    )

    if clamp_trick_training: # 能收敛
        pred_xstart_coord = torch.clamp(pred_xstart[:, :, 0:2], min=-1, max=1)
        pred_xstart_seman = pred_xstart[:, :, 2:] >= 0.5
        pred_xstart_cs = torch.cat((pred_xstart_coord, pred_xstart_seman), dim=2)
    else: # 不收敛
        pred_xstart_coord = pred_xstart[:, :, 0:2]
        pred_xstart_seman = pred_xstart[:, :, 2:]
        pred_xstart_cs = torch.cat((pred_xstart_coord, pred_xstart_seman), dim=2)

    # corners_loss_masked2 = (corners_withsemantics_target2 - pred_xstart_cs) ** 2

    # 局部/全局 对齐/重叠 的时间权重，batch(bs, 53, 2+7+1)内每个样本相同。t=0权重接近1，t=1000权重接近0
    # LACE原文写的是1-alphas_cumprod，写错了
    time_weight = torch.tensor(betas.tolist()[::-1], dtype=torch.float64, device=device)
    # 我们先求出局部对齐的loss，再乘以时间权重
    # print(pred_xstart_cs)

    # 根据最后一列是否为0来筛选出对应的行
    mask = pred_xstart_cs[:, :, -1] == 0

    # 仅选择最后一列为0的行的前两列（x和y坐标）
    x_coords = pred_xstart_cs[:, :, 0] * mask
    y_coords = pred_xstart_cs[:, :, 1] * mask

    # 设置mask中的False值为无穷大，以便在计算距离时排除这些值
    inf_mask = torch.where(mask, 0, float('inf'))
    x_coords += inf_mask
    y_coords += inf_mask

    # 计算 x 坐标的 L1 距离方阵
    x_coords_uns = x_coords.unsqueeze(2)
    distance_matrix_x = torch.abs(x_coords_uns - x_coords_uns.transpose(1, 2))

    # 计算 y 坐标的 L1 距离方阵
    y_coords_uns = y_coords.unsqueeze(2)
    distance_matrix_y = torch.abs(y_coords_uns - y_coords_uns.transpose(1, 2))

    # 设置无穷大和 NaN 值为99999
    distance_matrix_x[torch.isinf(distance_matrix_x) | torch.isnan(distance_matrix_x)] = 99999
    distance_matrix_y[torch.isinf(distance_matrix_y) | torch.isnan(distance_matrix_y)] = 99999
    # 对角线使用掩码填充99999
    distance_matrix_x[(torch.eye(53).unsqueeze(0) == 1).to(device).expand(distance_matrix_x.size(0), 53, 53)] = 99999
    distance_matrix_y[(torch.eye(53).unsqueeze(0) == 1).to(device).expand(distance_matrix_y.size(0), 53, 53)] = 99999
    # print(distance_matrix_x)
    # print(distance_matrix_y)
    # 计算每个节点 i 的所有 x, y 轴向距离的最小值
    min_values_x, _ = torch.min(distance_matrix_x, dim=2)
    min_values_y, _ = torch.min(distance_matrix_y, dim=2)
    # print(min_values_x.shape) # (bs, 53)
    # print(min_values_y.shape) # (bs, 53)
    # LACE原文：求同一个节点i对其他节点（53）的所有方向（2）中距离（106）最小的
    min_values = torch.stack((min_values_x, min_values_y), dim=2)
    min_values, _ = torch.min(min_values, dim=2)
    # print(min_values[29]) # (bs, 53) 包含99999,那代表节点为padding节点
    min_values_unnorm = torch.where(min_values != 99999, min_values * 128, torch.tensor(0.0, dtype=min_values.dtype, device=device))
    # print(min_values_unnorm[29])
    min_values_bin_masked = map_to_binary(min_values_unnorm)
    min_values_four_masked = map_to_fournary(min_values_unnorm)
    min_values_eight_masked = map_to_eightnary(min_values_unnorm)
    min_values_sxt_masked = map_to_sxtnary(min_values_unnorm)
    # print(min_values_bin_masked[29])
    # print(min_values_bin_masked[29].shape)






    # 对于每个样本，将非99999的元素套上g=-2log(1-(0.5-eps)x)
    # 应用函数 -2log(1-0.5x) 到非99999的元素
    # 注意：确保输入到log的值是正数，即1-0.5x > 0 # 这个x是距离所以一定大于等于0, 因为归一化L1距离一定小于等于2
    # 99999这里会直接变成0，没有影响
    # 再乘以时间权重
    # 12位按位回归，最大值是12
    # 不能干扰了主loss，无限进制下是[0, 255]->[0, 2]的距离，二进制这里变成了[0, 12]，我们给这个除以128，以与无限进制下的距离对齐
    masked_tensor_bin = ((-12 * torch.log(1 - (1 / 12 - 1e-8) * min_values_bin_masked.sum(2)).sum(1) * time_weight[
        t]).sum() / 128) / batch_size
    masked_tensor_four = ((-18 * torch.log(1 - (1 / 18 - 1e-8) * min_values_four_masked.sum(2)).sum(1) * time_weight[
        t]).sum() / 128) / batch_size
    masked_tensor_eight = ((-35 * torch.log(1 - (1 / 35 - 1e-8) * min_values_eight_masked.sum(2)).sum(1) * time_weight[
        t]).sum() / 128) / batch_size
    masked_tensor_sxt = ((-45 * torch.log(1 - (1 / 45 - 1e-8) * min_values_sxt_masked.sum(2)).sum(1) * time_weight[
        t]).sum() / 128) / batch_size
    masked_tensor_inf = (torch.where(min_values != 99999, -2 * torch.log(1 - (0.5 - 1e-8) * min_values),
                                torch.tensor(0.0, dtype=min_values.dtype, device=device)).sum(1) * time_weight[t]).sum() / batch_size
    # 对结果求和
    local_aligned_loss = masked_tensor_bin + masked_tensor_four + masked_tensor_eight + masked_tensor_sxt + masked_tensor_inf


    # 加权
    corner_number = torch.ones((batch_size,), device=device) * 53
    corners_loss_masked_avgpergraph1 = corners_loss_masked1.sum(dim=[1, 2]) / corner_number
    # corners_loss_masked_avgpergraph2 = corners_loss_masked2.sum(dim=[1, 2]) / corner_number
    corners_loss_batch1 = corners_loss_masked_avgpergraph1.mean()
    corners_loss_batch2 = 0
    corners_loss_batch1 *= 1
    # corners_loss_batch2 *= 1
    local_aligned_loss *= 1
    loss_batch = corners_loss_batch1 + corners_loss_batch2 + local_aligned_loss

    '''loss backward'''
    loss_batch.backward()
    '''clip norm, if False, training long steps may diverge'''
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    '''update params'''
    optimizer.step()
    '''step += 1'''
    step += 1


    '''print loss, loss save'''
    print('step: ', step,
          'loss * 100000: ', loss_batch.item() * 100000,
          'corner loss epsilon * 100000: ', corners_loss_batch1.item() * 100000,
          'local align loss * 100000: ', local_aligned_loss.item() * 100000,
          'bin * 1e5:', masked_tensor_bin.item() * 100000,
          'four * 1e5:', masked_tensor_four.item() * 100000,
          'eight * 1e5:', masked_tensor_eight.item() * 100000,
          'sxt * 1e5:', masked_tensor_sxt.item() * 100000,
          'inf * 1e5:', masked_tensor_inf.item() * 100000)
    loss_curve.append(
        [loss_batch.item() * 100000, corners_loss_batch1.item() * 100000, local_aligned_loss.item() * 100000,
         masked_tensor_bin.item() * 100000, masked_tensor_four.item() * 100000, masked_tensor_eight.item() * 100000,
         masked_tensor_sxt.item() * 100000,  masked_tensor_inf.item() * 100000])

    '''save model per interval steps'''
    if step % interval == 0:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            state_dict[name] = list(model.parameters())[i]
        torch.save(state_dict, output_dir + f"model{step:07d}.pt")
        torch.save(optimizer.state_dict(), output_dir + f"optim{step:07d}.pt")
        '''saving loss curve'''
        np.save(output_dir + 'loss_curve.npy', np.array(loss_curve))

    '''evaluate per interval steps'''
    if step % interval == 0 or step == 100 or step == 10000:
        model.eval()


        # results_timesteps_stage1_val = [999, 500, 250, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0, 'gt']
        results_timesteps_stage1_val = [0]
        results_stage1_val = {}  # 999/900/.../0/'gt': (results, results_adjacency_lists)
        for k_val in results_timesteps_stage1_val:
            results_stage1_val['results_corners_' + str(k_val)] = []
            results_stage1_val['results_semantics_' + str(k_val)] = []
            results_stage1_val['results_corners_numbers_' + str(k_val)] = []
        print('验证集一阶段开始')
        if len(dataset_val) % batch_size_val == 0:
            batch_numbers = len(dataset_val) // batch_size_val
        else:
            batch_numbers = len(dataset_val) // batch_size_val + 1
        for batch_count in tqdm(range(batch_numbers)):
            corners_withsemantics_0_val_batch, global_attn_matrix_val_batch, corners_padding_mask_val_batch, edges_val_batch = next(dataloader_val_iter)
            corners_withsemantics_0_val_batch = corners_withsemantics_0_val_batch.to(device).clamp(-1, 1)
            global_attn_matrix_val_batch = global_attn_matrix_val_batch.to(device)
            corners_padding_mask_val_batch = corners_padding_mask_val_batch.to(device)

            # (bs, 53, 10)
            corners_withsemantics_0_val_batch = torch.cat((corners_withsemantics_0_val_batch, (1 - corners_padding_mask_val_batch).type(corners_withsemantics_0_val_batch.dtype)), dim=2)

            '''reverse process: 999->998->...->1->0->final_pred(start)'''
            for current_step_val in list(range(diffusion_steps - 1, -1, -1)):
                if current_step_val == diffusion_steps - 1:
                    corners_withsemantics_t_val_batch = torch.randn(*corners_withsemantics_0_val_batch.shape, device=device,
                                                              dtype=corners_withsemantics_0_val_batch.dtype)  # t=999
                else:
                    corners_withsemantics_t_val_batch = sample_from_posterior_normal_distribution_val_batch

                t_val = torch.tensor([current_step_val] * 1, device=device)

                with torch.no_grad():
                    '''model: predicts corners_noise'''
                    output_corners_withsemantics1_val_batch, output_corners_withsemantics2_val_batch = model(
                        corners_withsemantics_t_val_batch, global_attn_matrix_val_batch, t_val)
                    output_corners_withsemantics_val_batch = torch.cat(
                        (output_corners_withsemantics1_val_batch, output_corners_withsemantics2_val_batch), dim=2)

                    '''gaussian posterior'''
                    model_variance_val_batch = torch.tensor(posterior_variance, device=device)[t_val][:, None,
                                         None].expand_as(
                        corners_withsemantics_t_val_batch)

                    pred_xstart_val_batch = (
                            torch.tensor(sqrt_recip_alphas_cumprod, device=device)[t_val][:, None, None].expand_as(
                                corners_withsemantics_t_val_batch) * corners_withsemantics_t_val_batch -
                            torch.tensor(sqrt_recipm1_alphas_cumprod, device=device)[t_val][:, None, None].expand_as(
                                corners_withsemantics_t_val_batch) * output_corners_withsemantics_val_batch
                    )  # to keep x0 always in (-1, 1) is unsuitable for which semantics number can be 2,3,...
                    pred_xstart_val_batch[:, :, 0:2] = torch.clamp(pred_xstart_val_batch[:, :, 0:2], min=-1, max=1)
                    pred_xstart_val_batch[:, :, 2:] = pred_xstart_val_batch[:, :, 2:] >= 0.5

                    model_mean_val_batch = (
                            torch.tensor(posterior_mean_coef1, device=device)[t_val][:, None, None].expand_as(
                                corners_withsemantics_t_val_batch) * pred_xstart_val_batch
                            + torch.tensor(posterior_mean_coef2, device=device)[t_val][:, None, None].expand_as(
                        corners_withsemantics_t_val_batch) * corners_withsemantics_t_val_batch
                    )
                    noise_val_batch = torch.randn_like(corners_withsemantics_t_val_batch)
                    sample_from_posterior_normal_distribution_val_batch = model_mean_val_batch + torch.sqrt(
                        model_variance_val_batch) * noise_val_batch
                    '''record to visualize'''
                    if current_step_val in results_timesteps_stage1_val:
                        for i in range(corners_withsemantics_0_val_batch.shape[0]):
                            results_stage1_val['results_corners_' + str(current_step_val)].append(
                                sample_from_posterior_normal_distribution_val_batch[i, :, :2][None, :, :])
                            results_stage1_val['results_semantics_' + str(current_step_val)].append(
                                sample_from_posterior_normal_distribution_val_batch[i, :, 2:15][None, :, :])
                            results_stage1_val['results_corners_numbers_' + str(current_step_val)].append(
                                sample_from_posterior_normal_distribution_val_batch[i, :, 15:16][None, :, :].view(-1))  # 0: True, 1: False




        '''inverse normalize, remove padding and rasterization'''
        stage1_0_val = None
        for k_val in results_timesteps_stage1_val:
            result_corners_inverse_normalized_val, result_semantics_inverse_normalized_val = \
                inverse_normalize_and_remove_padding_100(results_stage1_val['results_corners_' + str(k_val)],
                                                    results_stage1_val['results_semantics_' + str(k_val)],
                                                    results_stage1_val['results_corners_numbers_' + str(k_val)])
            stage1_0_val = (result_corners_inverse_normalized_val, result_semantics_inverse_normalized_val)

        # print(stage1_0_val)
        '''stage 2'''
        print('验证集二阶段开始')
        # merge points (data loading in actual)
        corners_all_samples_val = stage1_0_val[0]
        semantics_all_samples_val = stage1_0_val[1]
        if merge_points:
            corners_all_samples_merged_val = []
            semantics_all_samples_merged_val = []
            for i_val in range(len(dataset_val)):
                corners_i_val = corners_all_samples_val[i_val]
                semantics_i_val = semantics_all_samples_val[i_val]
                corners_merge_components_val = get_near_corners(corners_i_val, merge_threshold=0.01 * 256)
                indices_list_val = corners_merge_components_val
                corners_i_val = corners_i_val.reshape(-1, 2)
                semantics_i_val = semantics_i_val.reshape(-1, 13)
                full_indices_list_val = []
                for index_set_val in indices_list_val:
                    full_indices_list_val.extend(list(index_set_val))
                random_indices_list_val = []
                for index_set_val in indices_list_val:
                    random_index_val = random.choice(list(index_set_val))
                    random_indices_list_val.append(random_index_val)

                merged_corners_i_val = merge_array_elements(corners_i_val, full_indices_list_val, random_indices_list_val)
                merged_semantics_i_val = merge_array_elements(semantics_i_val, full_indices_list_val, random_indices_list_val)

                corners_all_samples_merged_val.append(merged_corners_i_val[None, :, :])
                semantics_all_samples_merged_val.append(merged_semantics_i_val[None, :, :])

            corners_all_samples_val = corners_all_samples_merged_val
            semantics_all_samples_val = semantics_all_samples_merged_val
        # print(corners_all_samples_val)
        # print(semantics_all_samples_val)

        # model 2 loading
        model_path_2 = 'outputs/structure-56-16-interval1000-lifull/' + 'model014000.pt' # 凑合一个就行，这个训完了还得换个更好的边缘模型
        model_2 = EdgeModel().to(device)
        model_2.load_state_dict(torch.load(model_path_2, map_location="cpu"))
        model_2.to(device)
        model_2.eval()


        results_timesteps_stage2_val = [0]
        results_stage2_val = {}
        for k_val in results_timesteps_stage2_val:
            results_stage2_val['results_corners_' + str(k_val)] = []
            results_stage2_val['results_edges_' + str(k_val)] = []
            results_stage2_val['results_corners_numbers_' + str(k_val)] = []

        for val_count in tqdm(range(len(dataset_val))):
            '''a batch of data'''
            corners_stage2_val = torch.zeros((1, 53, 2), dtype=torch.float64, device=device)
            corners_temp_stage2_val = (torch.tensor(corners_all_samples_val[val_count], dtype=torch.float64,
                                         device=device) - 128) / 128
            corners_stage2_val[:, 0:corners_temp_stage2_val.shape[1], :] = corners_temp_stage2_val

            semantics_stage2_val = torch.zeros((1, 53, 13), dtype=torch.float64, device=device)
            semantics_temp_stage2_val = torch.tensor(semantics_all_samples_val[val_count], dtype=torch.float64,
                                                    device=device)
            semantics_stage2_val[:, 0:semantics_temp_stage2_val.shape[1], :] = semantics_temp_stage2_val

            global_attn_matrix_stage2_val = torch.zeros((1, 53, 53), dtype=torch.bool, device=device)
            global_attn_matrix_stage2_val[:, 0:corners_temp_stage2_val.shape[1], 0:corners_temp_stage2_val.shape[1]] = True
            corners_padding_mask_stage2_val = torch.zeros((1, 53, 1), dtype=torch.uint8, device=device)
            corners_padding_mask_stage2_val[:, 0:corners_temp_stage2_val.shape[1], :] = 1

            # print(corners_stage2_val)
            # print(semantics_stage2_val)

            with torch.no_grad():
                '''model: Edge transformer'''
                output_edges_val = model_2(corners_stage2_val, global_attn_matrix_stage2_val, corners_padding_mask_stage2_val, semantics_stage2_val)
                output_edges_val = F.softmax(output_edges_val, dim=2)
                output_edges_val = torch.argmax(output_edges_val, dim=2)
                output_edges_val = F.one_hot(output_edges_val, num_classes=2)

                '''record to visualize'''
                results_stage2_val['results_corners_' + str(0)].append(corners_stage2_val)
                results_stage2_val['results_edges_' + str(0)].append(output_edges_val)
                results_stage2_val['results_corners_numbers_' + str(0)].append(torch.sum(corners_padding_mask_stage2_val.view(-1)).item())

        '''inverse normalize, remove padding and visualization'''
        for k_val in [0]:
            edges_all_samples_val = edges_remove_padding(results_stage2_val['results_edges_' + str(k_val)],
                                                     results_stage2_val['results_corners_numbers_' + str(k_val)])

            output_dir_val = output_dir + 'val_' + f'{step:07d}' + '/'
            if os.path.exists(output_dir_val):
                shutil.rmtree(output_dir_val)  # 删除路径
            os.makedirs(output_dir_val)  # 创建路径
            for val_count in tqdm(range(len(dataset_val))):
                corners_sample_i_val = corners_all_samples_val[val_count]
                edges_sample_i_val = edges_all_samples_val[val_count]
                semantics_sample_i_val = semantics_all_samples_val[val_count]

                # print(corners_sample_i_val.shape) # (1, 29, 2)
                # print(edges_sample_i_val.shape) # (1, 841, 2)
                # print(semantics_sample_i_val.shape) # (1, 29, 7)
                # print(corners_sample_i_val)
                # print(edges_sample_i_val)
                # print(semantics_sample_i_val)

                ''' get planar cycles'''
                # 形状为 (1, n, 14) 的 ndarray，包含 0 和 1;找到每个子数组中 1 所在的索引,用 99999 替换值为 0 的原始元素
                semantics_sample_i_transform_val = semantics_sample_i_val
                semantics_sample_i_transform_indices_val = np.indices(semantics_sample_i_transform_val.shape)[-1]
                semantics_sample_i_transform_val = np.where(semantics_sample_i_transform_val == 1,
                                                        semantics_sample_i_transform_indices_val, 99999)

                output_points_val = [tuple(corner_with_seman_val) for corner_with_seman_val in
                                 np.concatenate((corners_sample_i_val, semantics_sample_i_transform_val), axis=-1).tolist()[0]]
                output_edges_val = edges_to_coordinates(
                    np.triu(edges_sample_i_val[0, :, 1].reshape(len(output_points_val), len(output_points_val))).reshape(-1),
                    output_points_val)
                # print(output_points_val)
                # print(output_edges_val)

                d_rev_val, simple_cycles_val, simple_cycles_semantics_val = get_cycle_basis_and_semantic_3_semansimplified_lifull(output_points_val,
                                                                                               output_edges_val)
                # for k, v in d_rev_val.items():
                #     print(k, v)
                # for sc in simple_cycles_val:
                #     print(sc)
                # for scs in simple_cycles_semantics_val:
                #     print(scs)

                # 创建一个256x256的全白图片
                img = np.ones((256, 256, 3), np.uint8) * 255

                # draw
                for polygon_i, polygon in enumerate(simple_cycles_val):
                    cv2.fillPoly(img, [np.array([[p[0], p[1]] for p in polygon], dtype=np.int32)],
                                 color=colors[simple_cycles_semantics_val[polygon_i]])
                for polygon_i, polygon in enumerate(simple_cycles_val):
                    for point_i, point in enumerate(polygon):
                        if point_i < len(polygon) - 1:
                            p1 = (point[0], point[1])
                            cv2.rectangle(img, (p1[0] - 3, p1[1] - 3), (p1[0] + 3, p1[1] + 3), color=(150, 150, 150),
                                          thickness=-1)
                            p2 = (polygon[point_i + 1][0], polygon[point_i + 1][1])
                            cv2.line(img, p1, p2, color=(150, 150, 150), thickness=5)  # 这个地方设成5输出的才是7个像素宽

                cv2.imwrite(os.path.join(output_dir_val, f"val_pred_{val_count}.png"), img)

        print('验证集预测图渲染完毕')
        '''calculate FID, KID. '''
        current_Fid = fid(gt_dir_val, output_dir_val, fid_batch_size=128, fid_device=device)
        # current_Kid = kid(gt_dir_val, output_dir_val, kid_batch_size=128, kid_device=device)
        current_Kid = kid2(gt_dir_val, output_dir_val, kid_batch_size=128, kid_device=device)
        print('step: ', step, 'FID: ', current_Fid, 'KID: ', current_Kid)

        '''saving Acc'''
        val_metrics.append([step, current_Fid, current_Kid])
        np.save(output_dir + 'val_metrics.npy', np.array(val_metrics))

        model.train()

        '''lr decay'''
        if step in [500000]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
