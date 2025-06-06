import sys
sys.path.append('/home/user00/HSZ/gsdiff_boun-main')
sys.path.append('/home/user00/HSZ/gsdiff_boun-main/datasets')
sys.path.append('/home/user00/HSZ/gsdiff_boun-main/gsdiff_boun')

'''We found that calculating feature maps through CNN in real time consumes too much GPU memory, 
so after training CNN, the feature maps corresponding to the boundary images of the entire dataset are storaged in disk for reading.'''
'''Note: this script needs about 500GB disk space.'''

import math
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from datasets.rplang_edge_semantics_simplified_78_10_prerunCNN import RPlanGEdgeSemanSimplified_78_10_prerunCNN
from gsdiff_boun.boundary_78_10 import BoundaryModel
from gsdiff_boun.utils import *
from itertools import cycle
import torch.nn.functional as F
import numpy as np
import os


batch_size = 1
device = 'cuda:0' # Modify it yourself


'''Neural Network'''
pretrained_encoder = BoundaryModel().to(device)
pretrained_encoder.load_state_dict(torch.load('outputs/structure-78-12/model_stage0_best_006700.pt', map_location=device))
print('Pre-trained boundary CNN parameters：', sum(p.numel() for p in pretrained_encoder.parameters()))
for param in pretrained_encoder.parameters():
    param.requires_grad = False
print('CNN has been frozen')

'''Data'''
dataset_train = RPlanGEdgeSemanSimplified_78_10_prerunCNN('train', random_training_data=False)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=False)  # try different num_workers to be faster
dataloader_train_iter = iter(cycle(dataloader_train))
dataset_val = RPlanGEdgeSemanSimplified_78_10_prerunCNN('val', random_training_data=False)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=False)  # try different num_workers to be faster
dataloader_val_iter = iter(cycle(dataloader_val))
dataset_test = RPlanGEdgeSemanSimplified_78_10_prerunCNN('test', random_training_data=False)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=False)  # try different num_workers to be faster
dataloader_test_iter = iter(cycle(dataloader_test))

step = 0
while step < len(dataset_train): # 65763

    '''a batch of data'''
    normalized_image, file_id = next(dataloader_train_iter)
    normalized_image = normalized_image.to(device).float() # (bs=1, c=3, h=256, w=256)



   
    '''model'''
    feat = {}
    # output_images = pretrained_encoder(normalized_image)
    # print('x', x.shape) torch.Size([1, 3, 256, 256])
    e1 = pretrained_encoder.Conv1(normalized_image)
    # print('e1', e1.shape) torch.Size([1, 64, 256, 256])
    
    e2 = pretrained_encoder.Maxpool1(e1)
    # print('e2', e2.shape) torch.Size([1, 64, 128, 128])

    e2 = pretrained_encoder.Conv2(e2) + pretrained_encoder.shortcut2(e2)
    # print('e2', e2.shape) torch.Size([1, 128, 128, 128])

    e3 = pretrained_encoder.Maxpool2(e2)
    # print('e3', e3.shape) torch.Size([1, 128, 64, 64])
    e3 = pretrained_encoder.Conv3(e3) + pretrained_encoder.shortcut3(e3)
    feat[64] = e3.cpu().numpy()
    # print('e3', e3.shape) torch.Size([1, 256, 64, 64]) # 

    e4 = pretrained_encoder.Maxpool3(e3)
    # print('e4', e4.shape) torch.Size([1, 256, 32, 32])
    e4 = pretrained_encoder.Conv4(e4) + pretrained_encoder.shortcut4(e4)
    feat[32] = e4.cpu().numpy()
    # print('e4', e4.shape) torch.Size([1, 512, 32, 32])

    e5 = pretrained_encoder.Maxpool4(e4)
    # print('e5', e5.shape) torch.Size([1, 512, 16, 16])
    e5 = pretrained_encoder.Conv5(e5) + pretrained_encoder.shortcut5(e5)
    feat[16] = e5.cpu().numpy()
    # print('e5', e5.shape) torch.Size([1, 1024, 16, 16])

    # Save feature map
    np.save(os.path.join('../datasets/prerunning_cnn_featuremaps', str(file_id.item()) + '.npy'), feat)
    
    


    '''step += 1'''
    step += 1


    print('step: ', step, e5.shape, len(feat), feat[32].shape)



step = 0
while step < len(dataset_val): # 3000

    '''a batch of data'''
    normalized_image, file_id = next(dataloader_val_iter)
    normalized_image = normalized_image.to(device).float() # (bs=1, c=3, h=256, w=256)



   
    '''model'''
    feat = {}
    # output_images = pretrained_encoder(normalized_image)
    # print('x', x.shape) torch.Size([1, 3, 256, 256])
    e1 = pretrained_encoder.Conv1(normalized_image)
    # print('e1', e1.shape) torch.Size([1, 64, 256, 256])
    
    e2 = pretrained_encoder.Maxpool1(e1)
    # print('e2', e2.shape) torch.Size([1, 64, 128, 128])

    e2 = pretrained_encoder.Conv2(e2) + pretrained_encoder.shortcut2(e2)
    # print('e2', e2.shape) torch.Size([1, 128, 128, 128])

    e3 = pretrained_encoder.Maxpool2(e2)
    # print('e3', e3.shape) torch.Size([1, 128, 64, 64])
    e3 = pretrained_encoder.Conv3(e3) + pretrained_encoder.shortcut3(e3)
    feat[64] = e3.cpu().numpy()
    # print('e3', e3.shape) torch.Size([1, 256, 64, 64]) # 

    e4 = pretrained_encoder.Maxpool3(e3)
    # print('e4', e4.shape) torch.Size([1, 256, 32, 32])
    e4 = pretrained_encoder.Conv4(e4) + pretrained_encoder.shortcut4(e4)
    feat[32] = e4.cpu().numpy()
    # print('e4', e4.shape) torch.Size([1, 512, 32, 32])

    e5 = pretrained_encoder.Maxpool4(e4)
    # print('e5', e5.shape) torch.Size([1, 512, 16, 16])
    e5 = pretrained_encoder.Conv5(e5) + pretrained_encoder.shortcut5(e5)
    feat[16] = e5.cpu().numpy()
    # print('e5', e5.shape) torch.Size([1, 1024, 16, 16])

    
    # Save feature map
    np.save(os.path.join('../datasets/prerunning_cnn_featuremaps', str(file_id.item()) + '.npy'), feat)
    
    


    '''step += 1'''
    step += 1


    print('step: ', step, e5.shape, len(feat), feat[32].shape)




step = 0
while step < len(dataset_test): # 3000

    '''a batch of data'''
    normalized_image, file_id = next(dataloader_test_iter)
    normalized_image = normalized_image.to(device).float() # (bs=1, c=3, h=256, w=256)



   
    '''model'''
    feat = {}
    # output_images = pretrained_encoder(normalized_image)
    # print('x', x.shape) torch.Size([1, 3, 256, 256])
    e1 = pretrained_encoder.Conv1(normalized_image)
    # print('e1', e1.shape) torch.Size([1, 64, 256, 256])
    
    e2 = pretrained_encoder.Maxpool1(e1)
    # print('e2', e2.shape) torch.Size([1, 64, 128, 128])

    e2 = pretrained_encoder.Conv2(e2) + pretrained_encoder.shortcut2(e2)
    # print('e2', e2.shape) torch.Size([1, 128, 128, 128])

    e3 = pretrained_encoder.Maxpool2(e2)
    # print('e3', e3.shape) torch.Size([1, 128, 64, 64])
    e3 = pretrained_encoder.Conv3(e3) + pretrained_encoder.shortcut3(e3)
    feat[64] = e3.cpu().numpy()
    # print('e3', e3.shape) torch.Size([1, 256, 64, 64]) # 

    e4 = pretrained_encoder.Maxpool3(e3)
    # print('e4', e4.shape) torch.Size([1, 256, 32, 32])
    e4 = pretrained_encoder.Conv4(e4) + pretrained_encoder.shortcut4(e4)
    feat[32] = e4.cpu().numpy()
    # print('e4', e4.shape) torch.Size([1, 512, 32, 32])

    e5 = pretrained_encoder.Maxpool4(e4)
    # print('e5', e5.shape) torch.Size([1, 512, 16, 16])
    e5 = pretrained_encoder.Conv5(e5) + pretrained_encoder.shortcut5(e5)
    feat[16] = e5.cpu().numpy()
    # print('e5', e5.shape) torch.Size([1, 1024, 16, 16])

    
    # Save feature map
    np.save(os.path.join('../datasets/prerunning_cnn_featuremaps', str(file_id.item()) + '.npy'), feat)
    
    


    '''step += 1'''
    step += 1


    print('step: ', step, e5.shape, len(feat), feat[32].shape)
