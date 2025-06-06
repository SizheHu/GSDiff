import sys
sys.path.append('/home/user00/HSZ/gsdiff_boun-main')
sys.path.append('/home/user00/HSZ/gsdiff_boun-main/datasets')
sys.path.append('/home/user00/HSZ/gsdiff_boun-main/gsdiff_boun')

'''This script is the second step of training CNN. 
It loads the training model of the first step (structure-78-10) and continues training on random black polygons.
The difference between it and the model of the first step is that the training data of the first step is random colored polygons, 
while this step is random black polygons (closer to the real RPLAN boundary image)'''

import math
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
# from datasets.rplang_edge_semantics_simplified_78_10 import RPlanGEdgeSemanSimplified_78_10 # This part is slightly different from what I wrote at that time, but the version I revised now should be my actual practice at that time.
from datasets.rplang_edge_semantics_simplified_78_11 import RPlanGEdgeSemanSimplified_78_11 # This part is slightly different from what I wrote at that time, but the version I revised now should be my actual practice at that time.
from gsdiff_boun.boundary_78_10 import BoundaryModel
from gsdiff_boun.utils import *
from itertools import cycle
import torch.nn.functional as F

lr = 1e-4
weight_decay = 0
total_steps = float("inf") # 200000
batch_size = 16
device = 'cuda:1'
# device = 'cpu'

'''create output_dir'''
output_dir = 'outputs/structure-78-11/'
os.makedirs(output_dir, exist_ok=False)

'''Neural Network'''
model = BoundaryModel().to(device)
model.load_state_dict(torch.load('outputs/structure-78-10/model089000.pt', map_location=device))
print('total params:', sum(p.numel() for p in model.parameters()))

'''Data'''
# dataset_train = RPlanGEdgeSemanSimplified_78_10('train', random_training_data=True)# This part is slightly different from what I wrote at that time, but the version I revised now should be my actual practice at that time.
dataset_train = RPlanGEdgeSemanSimplified_78_11('train', random_training_data=True)# This part is slightly different from what I wrote at that time, but the version I revised now should be my actual practice at that time.
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                        drop_last=True, pin_memory=False)  # try different num_workers to be faster
dataloader_train_iter = iter(cycle(dataloader_train))
# dataset_val = RPlanGEdgeSemanSimplified_78_10('val', random_training_data=True)# This part is slightly different from what I wrote at that time, but the version I revised now should be my actual practice at that time.
dataset_val = RPlanGEdgeSemanSimplified_78_11('val', random_training_data=True)# This part is slightly different from what I wrote at that time, but the version I revised now should be my actual practice at that time.
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=False)  # try different num_workers to be faster
dataloader_val_iter = iter(cycle(dataloader_val))

'''Optim'''
optimizer = AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)
# optimizer = SGD(list(model.parameters()), lr=lr, momentum=0.9)
'''Training'''
step = 0
loss_curve = []
val_metrics = []

# lr reduce settings
lr_reduce_patience = 5
lr_reduce_count = 0
stop_patience = 20
stop_count = 0


best_L1_Loss = 99999999

interval = 1000


while step < total_steps:
    '''a batch of data'''
    normalized_image = next(dataloader_train_iter)
    normalized_image = normalized_image.to(device).float() # (bs, c=3, h=256, w=256)
    # print('input', normalized_image.min(), normalized_image.max())

    '''clear all params grad, to prepare for new computation'''
    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
    

    '''model'''
    output_images = model(normalized_image)
    # print('output', output_images.min(), output_images.max())
    # output_images = output_images.tanh()
    # print('output', output_images.min(), output_images.max())

    '''target'''
    images_target = normalized_image
    # print('gt', images_target.min(), images_target.max())

    '''calculate loss. '''
    loss = torch.abs(output_images - images_target).sum() / batch_size # L1loss

    '''loss backward'''
    loss.backward()
    '''clip norm, if False, training long steps may diverge'''
    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=0.1)
    '''update params'''
    optimizer.step()
    '''step += 1'''
    step += 1



    '''print loss, loss save'''
    print('step: ', step,
          'loss: ', loss.item())
    loss_curve.append([loss.item()])


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
        output_dir_val = output_dir + 'val_' + f'{step:07d}'
        if os.path.exists(output_dir_val):
            shutil.rmtree(output_dir_val)
        os.makedirs(output_dir_val)
        
        model.eval()
        val_number = len(dataset_val) # 3000
        L1Loss_val = 0
        for val_count in tqdm(range(val_number)):
            '''a batch of data, batch size = 1'''
            normalized_image_val = next(dataloader_val_iter)
            normalized_image_val = normalized_image_val.to(device).float() # (bs, c=3, h=256, w=256)
            with torch.no_grad():
                '''model'''
                output_images_val = model(normalized_image_val)[0]
                # print('output_images_val 1', output_images_val.min(), output_images_val.max())
                # output_images_val = output_images_val.tanh()
                # print('output_images_val 2', output_images_val.min(), output_images_val.max())


                output_images_val = (output_images_val * 128 + 128).clip(min=0, max=255).cpu().numpy().astype(np.uint8).transpose((1, 2, 0))
                # print('output_images_val 3', output_images_val.min(), output_images_val.max())
                cv2.imwrite(os.path.join(output_dir_val, f"val_pred_{val_count}.png"), output_images_val)
            
                '''target'''
                images_target_val = normalized_image_val[0]
            
                '''calculate loss. '''
                L1Loss_val += torch.abs(output_images - images_target).sum().item() # L1loss


        '''calculate F1. '''
        current_L1_Loss = L1Loss_val / val_number
        print('step: ', step, 'current_L1_Loss: ', current_L1_Loss)

        '''saving Acc'''
        val_metrics.append([step, current_L1_Loss])
        np.save(output_dir + 'val_metrics.npy', np.array(val_metrics))

        model.train()

        '''lr decay on val pleateu'''
        if current_L1_Loss <= best_L1_Loss:
            best_L1_Loss = current_L1_Loss
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
