'''
Demystifying MMD GANs. Authors:Mikołaj Bińkowski, Danica J. Sutherland, Michael Arbel, Arthur Gretton.

计算方法：
1 把gt和pred结果按照完全相同的方式进行渲染
2 把渲染的两组图片分别放进/images_path1 /images_path2
'''
from tqdm import tqdm
import os
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from pytorch_fid.inception import InceptionV3
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import sys


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def polynomial_mmd(features_generated, features_real, degree=3, gamma=None, coef0=1):
    kernel_xx = np.mean(polynomial_kernel(features_generated, features_generated, degree=degree, gamma=gamma, coef0=coef0))
    kernel_yy = np.mean(polynomial_kernel(features_real, features_real, degree=degree, gamma=gamma, coef0=coef0))
    kernel_xy = np.mean(polynomial_kernel(features_generated, features_real, degree=degree, gamma=gamma, coef0=coef0))
    mmd = kernel_xx + kernel_yy - 2 * kernel_xy
    return mmd


# path1 = '/home/myubt/Projects/house_diffusion-main/scripts/metrics/3-channel-semantics-128'
# path2 = '/home/myubt/Projects/house_diffusion-main/scripts/metrics/testimgs1'
# batch_size = 64
# device = 'cuda:0'
# dims = 2048
# block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
# model = InceptionV3([block_idx]).to(device).eval()
#
#
# files1 = [os.path.join(path1, fn) for fn in os.listdir(path1)]
# dataset1 = ImagePathDataset(files1, transforms=TF.ToTensor())
# dataloader1 = torch.utils.data.DataLoader(dataset1,
#                                          batch_size=batch_size,
#                                          shuffle=False,
#                                          drop_last=False,
#                                          num_workers=1)
# pred_arr1 = np.empty((len(files1), dims)) # np.ndarray([len(test set), 2048])
# start_idx1 = 0
# for batch1 in tqdm(dataloader1):
#     batch1 = batch1.to(device) # torch.Size([64, 3, 256, 256])
#     '''FID的计算器中，我们也是用了inception网络。
#     inception其实就是特征提取的网络，最后一层输出图像的类别。
#     不过我们会去除最后的全连接或者池化层，使得我们得到一个2048维度的特征。'''
#     with torch.no_grad():
#         pred1 = model(batch1)[0] # torch.Size([64, 2048, 1, 1])
#     pred1 = pred1.squeeze(3).squeeze(2).cpu().numpy() # np.ndarray([64, 2048])
#     pred_arr1[start_idx1:start_idx1 + batch_size] = pred1
#     start_idx1 = start_idx1 + batch_size
#
#
# ''' 对另一组图像做同样的操作 '''
# files2 = [os.path.join(path2, fn) for fn in os.listdir(path2)]
# dataset2 = ImagePathDataset(files2, transforms=TF.ToTensor())
# dataloader2 = torch.utils.data.DataLoader(dataset2,
#                                          batch_size=batch_size,
#                                          shuffle=False,
#                                          drop_last=False,
#                                          num_workers=1)
# pred_arr2 = np.empty((len(files2), dims))
# start_idx2 = 0
# for batch2 in tqdm(dataloader2):
#     batch2 = batch2.to(device)
#     with torch.no_grad():
#         pred2 = model(batch2)[0]
#     pred2 = pred2.squeeze(3).squeeze(2).cpu().numpy()
#     pred_arr2[start_idx2:start_idx2 + batch_size] = pred2
#     start_idx2 = start_idx2 + batch_size
#
# # 使用上面定义的函数计算核 MMD。此时，mmd 就是 KID 得分。
# mmd = polynomial_mmd(pred_arr1, pred_arr2)
# print("KID score:", mmd * 1000)

def kid2(path1, path2, kid_batch_size, kid_device):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(kid_device).eval()

    files1 = [os.path.join(path1, fn) for fn in os.listdir(path1)]
    dataset1 = ImagePathDataset(files1, transforms=TF.ToTensor())
    dataloader1 = torch.utils.data.DataLoader(dataset1,
                                              batch_size=kid_batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=1)
    pred_arr1 = np.empty((len(files1), dims))  # np.ndarray([len(test set), 2048])
    start_idx1 = 0
    for batch1 in tqdm(dataloader1):
        batch1 = batch1.to(kid_device)  # torch.Size([64, 3, 256, 256])
        '''FID的计算器中，我们也是用了inception网络。
        inception其实就是特征提取的网络，最后一层输出图像的类别。
        不过我们会去除最后的全连接或者池化层，使得我们得到一个2048维度的特征。'''
        with torch.no_grad():
            pred1 = model(batch1)[0]  # torch.Size([64, 2048, 1, 1])
        pred1 = pred1.squeeze(3).squeeze(2).cpu().numpy()  # np.ndarray([64, 2048])
        pred_arr1[start_idx1:start_idx1 + kid_batch_size] = pred1
        start_idx1 = start_idx1 + kid_batch_size

    ''' 对另一组图像做同样的操作 '''
    files2 = [os.path.join(path2, fn) for fn in os.listdir(path2)]
    dataset2 = ImagePathDataset(files2, transforms=TF.ToTensor())
    dataloader2 = torch.utils.data.DataLoader(dataset2,
                                              batch_size=kid_batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=1)
    pred_arr2 = np.empty((len(files2), dims))
    start_idx2 = 0
    for batch2 in tqdm(dataloader2):
        batch2 = batch2.to(kid_device)
        with torch.no_grad():
            pred2 = model(batch2)[0]
        pred2 = pred2.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr2[start_idx2:start_idx2 + kid_batch_size] = pred2
        start_idx2 = start_idx2 + kid_batch_size

    # 使用上面定义的函数计算核 MMD。此时，mmd 就是 KID 得分。
    mmd = polynomial_mmd(pred_arr1, pred_arr2)
    return mmd * 1000