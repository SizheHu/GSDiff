import sys
sys.path.append('/home/user00/HSZ/gsdiff-main') # Modify it yourself
sys.path.append('/home/user00/HSZ/gsdiff-main/datasets') # Modify it yourself
sys.path.append('/home/user00/HSZ/gsdiff-main/gsdiff') # Modify it yourself

'''This is the first script to train a Transformer as a topological graph autoencoder, pre-train with random graphs.'''

import math
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from datasets.rplang_bubble_diagram import RPlanGBubbleDiagram
from gsdiff.bubble_diagram_57_9 import *
from gsdiff.utils import *
from itertools import cycle


lr = 1e-4
weight_decay = 0
total_steps = float("inf") # 200000
batch_size = 2048
device = 'cuda:0'

'''create output_dir'''
output_dir = 'outputs/structure-57-13/'
os.makedirs(output_dir, exist_ok=False)


'''Neural Network'''
model = TopoGraphModel().to(device)
print('total params:', sum(p.numel() for p in model.parameters()))

'''Data'''
dataset_train = RPlanGBubbleDiagram('train')
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8,
                        drop_last=True, pin_memory=True)  # try different num_workers to be faster
dataloader_train_iter = iter(cycle(dataloader_train))
dataset_val = RPlanGBubbleDiagram('val')
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
best_S_Acc = 0
current_S_Acc = 0
best_E_Acc = 0
current_E_Acc = 0
interval = 1000
while step < total_steps:
    '''a batch of data'''
    semantics, adjacency_matrix, semantics_padding_mask, global_matrix, room_numbers = next(dataloader_train_iter)
    semantics = semantics.to(device)
    adjacency_matrix = adjacency_matrix.to(device)
    semantics_padding_mask = semantics_padding_mask.to(device)
    global_matrix = global_matrix.to(device)

    '''clear all params grad, to prepare for new computation'''
    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


    '''model'''
    output_semantics, output_edges = model(semantics, adjacency_matrix, semantics_padding_mask, global_matrix)
    # print(output_semantics.shape) # torch.Size([8, 8, 7])
    # print(output_edges.shape) # torch.Size([8, 64, 2])
    # print(room_numbers.shape) # torch.Size([8, 5])
    # print(semantics.shape) # torch.Size([8, 8, 7])
    # print(adjacency_matrix.shape) # torch.Size([8, 8, 8])


    '''target'''
    semantics_target = semantics
    output_semantics = output_semantics.reshape(-1, 7)
    semantics_target[:, :, 6] += 0.01
    semantics_target = torch.argmax(semantics_target, dim=2).reshape(-1)
    # print(semantics_target)
    '''calculate loss. '''
    semantics_CELoss = torch.nn.CrossEntropyLoss(reduction='none')(output_semantics, semantics_target)
    # print(semantics_CELoss.shape)
    semantics_loss_masked = semantics_CELoss * (semantics_padding_mask.reshape(-1))
    # print(semantics_loss_masked.shape)
    semantics_loss_batch = torch.sum(semantics_loss_masked) / torch.sum(semantics_padding_mask)
    # print(torch.sum(semantics_loss_masked), torch.sum(semantics_padding_mask))

    '''target'''
    edges_target = torch.cat((1 - adjacency_matrix[:, :, :, None], adjacency_matrix[:, :, :, None]), dim=3).reshape(*output_edges.shape)
    # print(edges_target.shape)
    output_edges = output_edges.reshape(-1, 2)
    # print(output_edges.shape)
    edges_target = edges_target[:, :, 1].reshape(-1)
    # print(edges_target)
    '''calculate loss. '''
    edges_CELoss = torch.nn.CrossEntropyLoss(reduction='none')(output_edges, edges_target)
    # print(edges_CELoss.shape)
    edges_padding_mask = global_matrix.reshape(batch_size, -1, 1).reshape(-1)
    # print(edges_padding_mask.shape)
    edges_loss_masked = edges_CELoss * edges_padding_mask
    # print(edges_loss_masked.shape)
    edges_loss_batch = torch.sum(edges_loss_masked) / torch.sum(edges_padding_mask)
    # print(edges_loss_batch.shape)

    semantics_loss_batch *= 1
    edges_loss_batch *= 1
    loss_batch = semantics_loss_batch + edges_loss_batch

    '''loss backward'''
    loss_batch.backward()
    '''clip norm, if False, training long steps may diverge'''
    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1) # 0.1 maybe also good?
    '''update params'''
    optimizer.step()
    '''step += 1'''
    step += 1



    '''print loss, loss save'''
    print('step: ', step,
          'loss * 100000: ', loss_batch.item() * 100000,
          'semantic loss * 100000: ', semantics_loss_batch.item() * 100000,
          'edge loss * 100000: ', edges_loss_batch.item() * 100000)
    loss_curve.append(
        [loss_batch.item() * 100000,
         semantics_loss_batch.item() * 100000,
         edges_loss_batch.item() * 100000])


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
        val_number = len(dataset_val) # 3000
        SN = 0
        ST = 0
        EN = 0
        ET = 0
        for val_count in tqdm(range(val_number)):
            '''a batch of data, batch size = 1'''
            semantics_val, adjacency_matrix_val, semantics_padding_mask_val, global_matrix_val, room_numbers_val = next(dataloader_val_iter)
            semantics_val = semantics_val.to(device)
            adjacency_matrix_val = adjacency_matrix_val.to(device)
            semantics_padding_mask_val = semantics_padding_mask_val.to(device)
            global_matrix_val = global_matrix_val.to(device)

            with torch.no_grad():
                '''model: transformer'''
                output_semantics_val, output_edges_val = model(semantics_val, adjacency_matrix_val, semantics_padding_mask_val, global_matrix_val)

                output_semantics_val = torch.argmax(F.softmax(output_semantics_val[:, :, :6], dim=2), dim=2)
                semantics_number_val = torch.sum(semantics_padding_mask_val).item()
                output_semantics_val = output_semantics_val.reshape(-1)[:semantics_number_val]

                output_edges_val = torch.argmax(F.softmax(output_edges_val, dim=2), dim=2).reshape(8, 8)[:semantics_number_val, :semantics_number_val].reshape(-1)

                '''target'''
                semantics_target_val = semantics_val
                semantics_target_val[:, :, 6] += 0.01
                semantics_target_val = torch.argmax(semantics_target_val, dim=2).reshape(-1)[:semantics_number_val]

                edges_target_val = adjacency_matrix_val.reshape(8, 8)[:semantics_number_val, :semantics_number_val].reshape(-1)


                '''record current sample metrics'''
                semantics_True_val = torch.eq(output_semantics_val, semantics_target_val).sum().item()
                SN += semantics_number_val
                ST += semantics_True_val

                edges_True_val = torch.eq(output_edges_val, edges_target_val).sum().item()
                EN += semantics_number_val ** 2
                ET += edges_True_val



        '''calculate Acc. '''
        current_S_Acc = ST / SN
        current_E_Acc = ET / EN
        print('step: ', step, 'semantics Acc: ', current_S_Acc, 'edges Acc: ', current_E_Acc)

        '''saving Acc'''
        val_metrics.append([step, current_S_Acc, current_E_Acc])
        np.save(output_dir + 'val_metrics.npy', np.array(val_metrics))

        model.train()

        '''lr decay on val pleateu'''
        if current_S_Acc >= best_S_Acc and current_E_Acc >= best_E_Acc:
            best_S_Acc = current_S_Acc
            best_E_Acc = current_E_Acc
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
