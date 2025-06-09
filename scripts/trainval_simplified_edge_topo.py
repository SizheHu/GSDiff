import sys

sys.path.append('/home/user00/HSZ/gsdiff-main')
sys.path.append('/home/user00/HSZ/gsdiff-main/datasets')
sys.path.append('/home/user00/HSZ/gsdiff-main/gsdiff')

import math
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from datasets.rplang_edge_semantics_simplified import RPlanGEdgeSemanSimplified
from gsdiff.house_nn3 import *
from gsdiff.utils import *
from itertools import cycle

'''This script is used to simply train an edge prediction model for the validation phase of the first phase topology constraint node generation model.'''

lr = 1e-4
weight_decay = 1e-5
total_steps = float("inf")  # 200000
batch_size = 8  # 8
device = 'cuda:0'

'''create output_dir'''
output_dir = 'outputs/structure-3/'
os.makedirs(output_dir, exist_ok=False)

'''Neural Network'''
model = EdgeModel().to(device)
print('total params:', sum(p.numel() for p in model.parameters()))

'''Data'''
dataset_train = RPlanGEdgeSemanSimplified('train')
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8,
                              drop_last=True, pin_memory=True)  # try different num_workers to be faster
dataloader_train_iter = iter(cycle(dataloader_train))
dataset_val = RPlanGEdgeSemanSimplified('val')
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0,
                            drop_last=False, pin_memory=True)  # try different num_workers to be faster
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
interval = 100

from scipy.stats import truncnorm


def truncated_normal(tensor, mu, sigma, lower, upper, dtype, device):
    # Generates truncated Gaussian distribution samples of the same shape as the target tensor
    with torch.no_grad():
        size = tensor.shape
        tmp = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=size)
        tmp = torch.as_tensor(tmp, dtype=dtype, device=device)
        tensor.copy_(tmp)
    return tensor


while step < total_steps:
    '''a batch of data'''
    corners_withsemantics, global_attn_matrix, corners_padding_mask, edges = next(dataloader_train_iter)
    corners_withsemantics = corners_withsemantics.to(device).clamp(-1, 1)
    global_attn_matrix = global_attn_matrix.to(device)
    # print(global_attn_matrix.dtype) # torch.bool
    # assert 0
    corners_padding_mask = corners_padding_mask.to(device)

    edges = edges.to(device)

    corners = corners_withsemantics[:, :, :2]
    semantics = corners_withsemantics[:, :, 2:]

    # For the first n points, i.e. non-padding points, each point is randomly missing with a probability of 1%.
    # How to do it: Consider all our data:
    # corners(bs, 53, 2)，semantics(bs, 53, 7)，
    # global_attn_matrix(bs, 53, 53) is the matrix with 1 in the upper left corner.
    # corners_padding_mask(bs, 53, 1) is 1 on top and 0 on bottom.
    # edges(bs, 2809, 1)<==>(bs, 53, 53) is a matrix that is 1 only when edges exist
    # We consider randomly transforming [corners_padding_mask(bs, 53, 1) to 1 on top and 0 on bottom], and process the remaining data according to the subsequent transformation results
    '''padding_one_hot = torch.cat((1 - corners_padding_mask, corners_padding_mask), dim=2).reshape(batch_size * 53, 2,
                                                                                                 1).to(torch.float64)
    q_padding_one_hot = torch.tensor([[1, 0.01],
                                      [0, 0.99]], dtype=torch.float64, device=device)[None, None, :, :] \
        .expand(batch_size, 53, 2, 2).reshape(batch_size * 53, 2, 2)
    padding_prob = torch.bmm(q_padding_one_hot, padding_one_hot).reshape(batch_size * 53, 2)
    padding_sample = torch.multinomial(padding_prob, num_samples=1)
    corners_padding_mask = padding_sample.reshape(batch_size, 53, 1).to(corners_padding_mask.dtype)
    corners = corners * corners_padding_mask.expand(batch_size, 53, 2)
    semantics = semantics * corners_padding_mask.expand(batch_size, 53, 7)
    # global_attn_matrix(bs, 53, 53)
    global_attn_matrix = torch.logical_and(corners_padding_mask.to(torch.bool).expand(batch_size, 53, 53),
                                           corners_padding_mask.to(torch.bool).expand(batch_size, 53, 53).transpose(1,
                                                                                                                    2))
    # edges(bs, 2809, 1) = edges(bs, 2809, 1) and global_attn_matrix(bs, 53, 53)
    edges = torch.logical_and(edges.to(torch.bool).reshape(batch_size, 53, 53), global_attn_matrix).to(
        torch.uint8).reshape(batch_size, 2809, 1)'''

    '''padding2_one_hot = torch.cat((1 - corners_padding_mask, corners_padding_mask), dim=2).reshape(batch_size * 53, 2,
                                                                                                  1).to(torch.float64)
    q_padding2_one_hot = torch.tensor([[0.99, 0],
                                       [0.01, 1]], dtype=torch.float64, device=device)[None, None, :, :] \
        .expand(batch_size, 53, 2, 2).reshape(batch_size * 53, 2, 2)
    padding2_prob = torch.bmm(q_padding2_one_hot, padding2_one_hot).reshape(batch_size * 53, 2)
    padding2_sample = torch.multinomial(padding2_prob, num_samples=1)
    corners_padding2_mask = padding2_sample.reshape(batch_size, 53, 1).to(corners_padding_mask.dtype)
    noisy_corners_padding_mask = corners_padding2_mask - corners_padding_mask
    corners = corners + truncated_normal(torch.empty((batch_size, 53, 2), dtype=corners.dtype, device=corners.device),
                                     0, 1, -1, 1, dtype=corners.dtype, device=corners.device) * noisy_corners_padding_mask.expand(batch_size, 53, 2)
    semantics = semantics + torch.randint(low=0, high=1+1, size=semantics.shape, dtype=semantics.dtype, device=semantics.device) * noisy_corners_padding_mask.expand(batch_size, 53, 7)
    # global_attn_matrix(bs, 53, 53)
    corners_padding_mask = corners_padding2_mask
    global_attn_matrix = torch.logical_and(corners_padding_mask.to(torch.bool).expand(batch_size, 53, 53),
                                           corners_padding_mask.to(torch.bool).expand(batch_size, 53, 53).transpose(1, 2))'''
    # print(global_attn_matrix.to(torch.uint8))

    mu = 0
    sigma = 1 / 128
    lower = -3 * sigma
    upper = 3 * sigma
    corners_noise = truncated_normal(torch.empty((batch_size, 53, 2), dtype=corners.dtype, device=corners.device),
                                     mu, sigma, lower, upper, dtype=corners.dtype, device=corners.device)
    corners = corners + corners_noise  # Truncated Gaussian noise -> The standard deviation corresponds to the same normalization multiple as the corner points

    # Introduce some perturbations to the 7-dimensional semantics and randomly reverse each semantics (1% probability for each)
    '''semantics_one_hot = torch.stack((1 - semantics, semantics), dim=3).reshape(batch_size * 53 * 7, 2, 1)
    q_semantics_one_hot = torch.tensor([[0.99, 0.01],
                                        [0.01, 0.99]], dtype=semantics_one_hot.dtype, device=semantics_one_hot.device)[
                          None, None, None, :, :].expand(batch_size, 53, 7, 2, 2).reshape(batch_size * 53 * 7, 2, 2)
    semantics_prob = torch.bmm(q_semantics_one_hot, semantics_one_hot).reshape(batch_size * 53 * 7, 2)
    semantics_sample = torch.multinomial(semantics_prob, num_samples=1)
    semantics = semantics_sample.reshape(batch_size, 53, 7).to(corners_withsemantics.dtype)
    semantics = semantics * corners_padding_mask.expand(batch_size, 53, 7)'''

    edges = torch.cat((1 - edges, edges), dim=2).type(torch.uint8)

    '''clear all params grad, to prepare for new computation'''
    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    '''model: Edge transformer'''
    output_edges = model(corners, global_attn_matrix, corners_padding_mask, semantics)

    '''target'''
    edges_target = edges.reshape(*output_edges.shape)
    output_edges = output_edges.reshape(-1, 2)
    edges_target = edges_target[:, :, 1].reshape(-1)

    '''calculate loss. '''
    edges_CELoss = torch.nn.CrossEntropyLoss(reduction='none')(output_edges, edges_target)

    edges_padding_mask = global_attn_matrix.reshape(batch_size, -1, 1).type(torch.uint8).reshape(-1)
    edges_loss_masked = edges_CELoss * edges_padding_mask
    edges_loss_batch = torch.sum(edges_loss_masked) / torch.sum(edges_padding_mask)

    edges_loss_batch *= 1
    loss_batch = edges_loss_batch

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
          'edge loss * 100000: ', edges_loss_batch.item() * 100000)
    loss_curve.append(
        [loss_batch.item() * 100000, edges_loss_batch.item() * 100000])

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
            corners_withsemantics_val, global_attn_matrix_val, corners_padding_mask_val, edges_val = \
                next(dataloader_val_iter)
            corners_withsemantics_val = corners_withsemantics_val.to(device).clamp(-1, 1)
            corners_val = corners_withsemantics_val[:, :, :2]
            semantics_val = corners_withsemantics_val[:, :, 2:]
            corners_val = corners_val + (torch.randn_like(corners_val, dtype=corners_val.dtype,
                                                          device=corners_val.device) * 1 / 128)
            global_attn_matrix_val = global_attn_matrix_val.to(device)
            corners_padding_mask_val = corners_padding_mask_val.to(device)
            edges_val = edges_val.to(device).type(torch.uint8)
            with torch.no_grad():
                '''model: Edge transformer'''
                output_edges_val = model(corners_val, global_attn_matrix_val, corners_padding_mask_val, semantics_val)
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
