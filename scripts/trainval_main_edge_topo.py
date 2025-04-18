import sys

sys.path.append('/home/user00/HSZ/gsdiff_topo-main')
sys.path.append('/home/user00/HSZ/gsdiff_topo-main/datasets')
sys.path.append('/home/user00/HSZ/gsdiff_topo-main/gsdiff_topo')

import math
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from gsdiff_topo.heterhouse_56_31 import *
from gsdiff_topo.utils import *
from itertools import cycle

from gsdiff_topo.bubble_diagram_57_9 import TopoGraphModel

# 数据集我们把rplang_bubble_diagram_57_15和rplang_edge_semantics_simplified合起来
from datasets.rplang_edge_semantics_simplified_56_31 import RPlanGEdgeSemanSimplified_56_31

lr = 1e-4
weight_decay = 1e-5
total_steps = float("inf")  # 200000
batch_size = 8  # 8
device = 'cuda:0'

'''create output_dir'''
output_dir = 'outputs/structure-56-35/'
os.makedirs(output_dir, exist_ok=False)
'''record description'''
description = '''
weight decay 1e-5
depth 12
dim 256
坐标语义各128维concat
坐标有截尾高斯噪声
lambda辅助任务

56-35：56-30（最优边缘模型）带有拓扑图约束（56-31）去掉sigmoid
'''
file_description = open(output_dir + 'file_description.txt', mode='w')
file_description.write(description)
file_description.close()

'''Neural Network'''
model = TopoEdgeModel().to(device)
print('total params:', sum(p.numel() for p in model.parameters()))

'''Data'''
dataset_train = RPlanGEdgeSemanSimplified_56_31('train')
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=True, pin_memory=False)  # try different num_workers to be faster
dataloader_train_iter = iter(cycle(dataloader_train))
dataset_val = RPlanGEdgeSemanSimplified_56_31('val')
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0,
                            drop_last=False, pin_memory=False)  # try different num_workers to be faster
dataloader_val_iter = iter(cycle(dataloader_val))

'''Optim'''
optimizer = AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)

'''Training'''
step = 0
loss_curve = []
val_metrics = []

# lr reduce settings
lr_reduce_patience = 5
lr_reduce_count = 0
stop_patience = 20
stop_count = 0
best_Acc = 0
current_Acc = 0
interval = 1000

from scipy.stats import truncnorm


def truncated_normal(tensor, mu, sigma, lower, upper, dtype, device):
    # 生成与目标张量相同形状的截尾高斯分布样本
    with torch.no_grad():
        size = tensor.shape
        tmp = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=size)
        tmp = torch.as_tensor(tmp, dtype=dtype, device=device)
        tensor.copy_(tmp)
    return tensor









'''Neural Network'''
pretrained_encoder = TopoGraphModel().to(device)
pretrained_encoder.load_state_dict(torch.load('outputs/structure-57-16/model_stage0_best_006000.pt', map_location=device))
print('预训练图嵌入网络参数量：', sum(p.numel() for p in pretrained_encoder.parameters()))
# 冻结预训练嵌入网络参数
for param in pretrained_encoder.parameters():
    param.requires_grad = False
print('已冻结图嵌入网络')











while step < total_steps:
    '''a batch of data'''
    bb_semantics, bb_adjacency_matrix, bb_semantics_padding_mask, bb_global_matrix, bb_room_number, corners_withsemantics, global_attn_matrix, corners_padding_mask, edges = next(dataloader_train_iter)




    bb_semantics = bb_semantics.to(device)
    bb_adjacency_matrix = bb_adjacency_matrix.to(device)
    bb_semantics_padding_mask = bb_semantics_padding_mask.to(device)
    bb_global_matrix = bb_global_matrix.to(device)


    

    
    corners_withsemantics = corners_withsemantics.to(device).clamp(-1, 1)
    global_attn_matrix = global_attn_matrix.to(device)
    # print(global_attn_matrix.dtype) # torch.bool
    # assert 0
    corners_padding_mask = corners_padding_mask.to(device)

    edges = edges.to(device)

    corners = corners_withsemantics[:, :, :2]
    semantics = corners_withsemantics[:, :, 2:]

    # 对于前n个点，即非padding的点，每个点以1%的概率随机缺失，
    # 实现方法：考虑我们的所有数据：
    # corners(bs, 53, 2)，semantics(bs, 53, 7)，
    # global_attn_matrix(bs, 53, 53)为左上角1的矩阵，
    # corners_padding_mask(bs, 53, 1)为上1下0，
    # edges(bs, 2809, 1)<==>(bs, 53, 53)为只有边存在时才为1的矩阵
    # 我们考虑将【corners_padding_mask(bs, 53, 1)为上1下0】进行随机变换，根据之后的变换结果处理剩下的几个数据
    '''padding_one_hot = torch.cat((1 - corners_padding_mask, corners_padding_mask), dim=2).reshape(batch_size * 53, 2,
                                                                                                 1).to(torch.float64)
    q_padding_one_hot = torch.tensor([[1, 0.01],
                                      [0, 0.99]], dtype=torch.float64, device=device)[None, None, :, :] \
        .expand(batch_size, 53, 2, 2).reshape(batch_size * 53, 2, 2)
    padding_prob = torch.bmm(q_padding_one_hot, padding_one_hot).reshape(batch_size * 53, 2)
    padding_sample = torch.multinomial(padding_prob, num_samples=1)
    corners_padding_mask = padding_sample.reshape(batch_size, 53, 1).to(corners_padding_mask.dtype)
    # 重新处理corners，semantics，padding部分置0
    corners = corners * corners_padding_mask.expand(batch_size, 53, 2)  # padding只能是0
    semantics = semantics * corners_padding_mask.expand(batch_size, 53, 7)  # padding只能是0
    # global_attn_matrix(bs, 53, 53)
    global_attn_matrix = torch.logical_and(corners_padding_mask.to(torch.bool).expand(batch_size, 53, 53),
                                           corners_padding_mask.to(torch.bool).expand(batch_size, 53, 53).transpose(1,
                                                                                                                    2))
    # edges(bs, 2809, 1) = edges(bs, 2809, 1) and global_attn_matrix(bs, 53, 53)
    edges = torch.logical_and(edges.to(torch.bool).reshape(batch_size, 53, 53), global_attn_matrix).to(
        torch.uint8).reshape(batch_size, 2809, 1)'''

    # 对于padding的点，每个点以1%的概率随机变成噪点，
    # 实现方法：和随机缺失一样的
    '''padding2_one_hot = torch.cat((1 - corners_padding_mask, corners_padding_mask), dim=2).reshape(batch_size * 53, 2,
                                                                                                  1).to(torch.float64)
    q_padding2_one_hot = torch.tensor([[0.99, 0],
                                       [0.01, 1]], dtype=torch.float64, device=device)[None, None, :, :] \
        .expand(batch_size, 53, 2, 2).reshape(batch_size * 53, 2, 2)
    padding2_prob = torch.bmm(q_padding2_one_hot, padding2_one_hot).reshape(batch_size * 53, 2)
    padding2_sample = torch.multinomial(padding2_prob, num_samples=1)
    corners_padding2_mask = padding2_sample.reshape(batch_size, 53, 1).to(corners_padding_mask.dtype)
    noisy_corners_padding_mask = corners_padding2_mask - corners_padding_mask
    # 重新处理corners，semantics，padding部分置0
    # corners生成截尾标准高斯分布，然后乘以噪声掩膜noisy_corners_padding_mask，即可得到噪声坐标，加到corners即可
    # 语义可以生成均匀取值0或1；同样需要乘以噪声掩膜noisy_corners_padding_mask，然后加到semantics
    corners = corners + truncated_normal(torch.empty((batch_size, 53, 2), dtype=corners.dtype, device=corners.device),
                                     0, 1, -1, 1, dtype=corners.dtype, device=corners.device) * noisy_corners_padding_mask.expand(batch_size, 53, 2)
    semantics = semantics + torch.randint(low=0, high=1+1, size=semantics.shape, dtype=semantics.dtype, device=semantics.device) * noisy_corners_padding_mask.expand(batch_size, 53, 7)
    # global_attn_matrix(bs, 53, 53)
    corners_padding_mask = corners_padding2_mask
    global_attn_matrix = torch.logical_and(corners_padding_mask.to(torch.bool).expand(batch_size, 53, 53),
                                           corners_padding_mask.to(torch.bool).expand(batch_size, 53, 53).transpose(1, 2))'''
    # print(global_attn_matrix.to(torch.uint8)) # 验证正确
    # 边相比于随机删除之后的，只是增加了噪点，没有带来新的边，所以不变

    # 指定参数
    mu = 0
    sigma = 1 / 128
    lower = -3 * sigma
    upper = 3 * sigma
    corners_noise = truncated_normal(torch.empty((batch_size, 53, 2), dtype=corners.dtype, device=corners.device),
                                     mu, sigma, lower, upper, dtype=corners.dtype, device=corners.device)
    corners = corners + corners_noise  # 截尾高斯噪声 -> 和角点相同的归一化倍数对应标准差

    # 在56-3的基础上引入一些对7维语义的扰动，对每个语义随机反转(每一种1%概率)
    semantics_one_hot = torch.stack((1 - semantics, semantics), dim=3).reshape(batch_size * 53 * 7, 2, 1)
    q_semantics_one_hot = torch.tensor([[0.99, 0.01],
                                        [0.01, 0.99]], dtype=semantics_one_hot.dtype, device=semantics_one_hot.device)[
                          None, None, None, :, :].expand(batch_size, 53, 7, 2, 2).reshape(batch_size * 53 * 7, 2, 2)
    semantics_prob = torch.bmm(q_semantics_one_hot, semantics_one_hot).reshape(batch_size * 53 * 7, 2)
    semantics_sample = torch.multinomial(semantics_prob, num_samples=1)
    semantics = semantics_sample.reshape(batch_size, 53, 7).to(corners_withsemantics.dtype)
    semantics = semantics * corners_padding_mask.expand(batch_size, 53, 7)  # padding只能是0

    edges = torch.cat((1 - edges, edges), dim=2).type(torch.uint8)

    '''clear all params grad, to prepare for new computation'''
    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()








    '''给出训练集拓扑图作为约束，拓扑图经过图嵌入网络获得拓扑图节点的嵌入，作为约束。
        注意其中有padding，被mask成了全0。'''
    bb_semantics = bb_semantics.float()
    bb_semantics_embedding = pretrained_encoder.semantics_embedding(bb_semantics)
    for layer in pretrained_encoder.transformer_layers:
        bb_semantics_embedding = layer(bb_semantics_embedding, bb_adjacency_matrix)
    bb_semantics_embedding = bb_semantics_embedding * bb_semantics_padding_mask # (bs, 8, 256)











    

    '''model: Edge transformer'''
    output_edges, random_lambda, output_lambdas = model(corners, global_attn_matrix, corners_padding_mask, semantics, bb_semantics_embedding, bb_semantics_padding_mask)

    '''target'''
    edges_target = edges.reshape(*output_edges.shape)
    output_edges = output_edges.reshape(-1, 2)
    edges_target = edges_target[:, :, 1].reshape(-1)


    '''calculate loss. '''
    edges_CELoss = torch.nn.CrossEntropyLoss(reduction='none')(output_edges, edges_target)

    edges_padding_mask = global_attn_matrix.reshape(batch_size, -1, 1).type(torch.uint8).reshape(-1)
    edges_loss_masked = edges_CELoss * edges_padding_mask
    edges_loss_batch = torch.sum(edges_loss_masked) / torch.sum(edges_padding_mask)



    # output_lambdas = output_lambdas.sigmoid()
    output_lambdas = torch.where(output_lambdas > 0.5, 1 - output_lambdas, output_lambdas)
    random_lambda[random_lambda > 0.5] = 1 - random_lambda[random_lambda > 0.5]
    lambdas_loss_masked = torch.abs(random_lambda - output_lambdas) * edges_padding_mask.reshape(batch_size, -1, 1) # L1
    lambdas_loss_batch = torch.sum(lambdas_loss_masked) / torch.sum(edges_padding_mask)

    lambdas_loss_batch *= 1




    edges_loss_batch *= 1
    loss_batch = edges_loss_batch + lambdas_loss_batch

    '''loss backward'''
    loss_batch.backward()
    '''clip norm, if False, training long steps may diverge'''
    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=0.1)
    '''update params'''
    optimizer.step()
    '''step += 1'''
    step += 1

    '''print loss, loss save'''
    print('step: ', step,
          'loss * 100000: ', loss_batch.item() * 100000,
          'edge loss * 100000: ', edges_loss_batch.item() * 100000,
          'lambda loss * 100000: ', lambdas_loss_batch.item() * 100000)
    loss_curve.append(
        [loss_batch.item() * 100000, edges_loss_batch.item() * 100000, lambdas_loss_batch.item() * 100000])

    '''save model per interval steps'''
    if step % interval == 0:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            state_dict[name] = list(model.parameters())[i]
        torch.save(state_dict, output_dir + f"model{step:06d}.pt")
        torch.save(optimizer.state_dict(), output_dir + f"optim{step:06d}.pt")
        '''saving loss curve'''
        np.save(output_dir + 'loss_curve.npy', np.array(loss_curve))

    '''evaluate per interval steps'''
    if step % interval == 0:
        model.eval()
        val_number = len(dataset_val)  # 3000
        val_count = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for val_count in tqdm(range(val_number)):
            '''a batch of data, batch size = 1'''
            bb_semantics_val_batch, bb_adjacency_matrix_val_batch, bb_semantics_padding_mask_val_batch, bb_global_matrix_val_batch, bb_room_number_val_batch, corners_withsemantics_val, global_attn_matrix_val, corners_padding_mask_val, edges_val = \
                next(dataloader_val_iter)





            

            bb_semantics_val_batch = bb_semantics_val_batch.to(device) # padding到8的低维房间语义（房间语义和房间类型是一码事，毕竟不是自然图像）
            bb_adjacency_matrix_val_batch = bb_adjacency_matrix_val_batch.to(device) # 拓扑图的邻接矩阵
            bb_semantics_padding_mask_val_batch = bb_semantics_padding_mask_val_batch.to(device) # 拓扑图的padding mask
            bb_global_matrix_val_batch = bb_global_matrix_val_batch.to(device)# 拓扑图的global_attn_matrix


            '''给出训练集拓扑图作为约束，拓扑图经过图嵌入网络获得拓扑图节点的嵌入，作为约束。
            注意其中有padding，被mask成了全0。 (batch_size_val, 8, 256)
            注意我们在推断时encoder输出不变，所以不是运行1000次而是只把encoder运行1次，这就是标准的classifier guidance'''
            bb_semantics_val_batch = bb_semantics_val_batch.float()
            bb_semantics_embedding_val_batch = pretrained_encoder.semantics_embedding(bb_semantics_val_batch)
            for layer in pretrained_encoder.transformer_layers:
                bb_semantics_embedding_val_batch = layer(bb_semantics_embedding_val_batch, bb_adjacency_matrix_val_batch)
            bb_semantics_embedding_val_batch = bb_semantics_embedding_val_batch * bb_semantics_padding_mask_val_batch  # (bs, 8, 256)







            
            
            corners_withsemantics_val = corners_withsemantics_val.to(device).clamp(-1, 1)
            corners_val = corners_withsemantics_val[:, :, :2]
            semantics_val = corners_withsemantics_val[:, :, 2:]
            corners_val = corners_val + (torch.randn_like(corners_val, dtype=corners_val.dtype,
                                                          device=corners_val.device) * 1 / 128)  # 标准高斯噪声 -> 和角点相同的归一化倍数对应标准差
            global_attn_matrix_val = global_attn_matrix_val.to(device)
            corners_padding_mask_val = corners_padding_mask_val.to(device)
            edges_val = edges_val.to(device).type(torch.uint8)
            with torch.no_grad():
                '''model: Edge transformer'''




                
                output_edges_val, _, _ = model(corners_val, global_attn_matrix_val, corners_padding_mask_val, semantics_val, bb_semantics_embedding_val_batch, bb_semantics_padding_mask_val_batch)




                
                output_edges_val = F.softmax(output_edges_val, dim=2)
                output_edges_val = torch.argmax(output_edges_val, dim=2)
                '''target'''
                edges_target_val = edges_val.reshape(*output_edges_val.shape)
                valid_elements = global_attn_matrix_val.reshape(*output_edges_val.shape)
                '''record current sample metrics'''
                TP_item = torch.sum((output_edges_val == 1) & (edges_target_val == 1) & valid_elements).item()
                TN_item = torch.sum((output_edges_val == 0) & (edges_target_val == 0) & valid_elements).item()
                FP_item = torch.sum((output_edges_val == 1) & (edges_target_val == 0) & valid_elements).item()
                FN_item = torch.sum((output_edges_val == 0) & (edges_target_val == 1) & valid_elements).item()
                TP += TP_item
                TN += TN_item
                FP += FP_item
                FN += FN_item
        '''calculate Acc. '''
        current_Acc = (TP + TN) / (TP + TN + FP + FN)
        print('step: ', step, 'Acc: ', current_Acc)

        '''saving Acc'''
        val_metrics.append([step, current_Acc])
        np.save(output_dir + 'val_metrics.npy', np.array(val_metrics))

        model.train()

        '''lr decay on val pleateu'''
        if current_Acc >= best_Acc:
            best_Acc = current_Acc
            lr_reduce_count = 0
            stop_count = 0
        else:
            lr_reduce_count += 1
            stop_count += 1
        if stop_count >= stop_patience:
            sys.exit(1)
        if lr_reduce_count >= lr_reduce_patience:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            lr_reduce_count = 0