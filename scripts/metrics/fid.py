'''
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp
Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local
nash equilibrium. Advances in neural information processing systems 30 (2017).

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

def fid(path1, path2, fid_batch_size, fid_device):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(fid_device).eval()

    files1 = [os.path.join(path1, fn) for fn in os.listdir(path1)]
    dataset1 = ImagePathDataset(files1, transforms=TF.ToTensor())
    dataloader1 = torch.utils.data.DataLoader(dataset1,
                                             batch_size=fid_batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=1)
    pred_arr1 = np.empty((len(files1), dims)) # np.ndarray([len(test set), 2048])
    start_idx1 = 0
    for batch1 in tqdm(dataloader1):
        batch1 = batch1.to(fid_device) # torch.Size([64, 3, 256, 256])
        '''FID的计算器中，我们也是用了inception网络。
        inception其实就是特征提取的网络，最后一层输出图像的类别。
        不过我们会去除最后的全连接或者池化层，使得我们得到一个2048维度的特征。'''
        with torch.no_grad():
            pred1 = model(batch1)[0] # torch.Size([64, 2048, 1, 1])
        pred1 = pred1.squeeze(3).squeeze(2).cpu().numpy() # np.ndarray([64, 2048])
        pred_arr1[start_idx1:start_idx1 + fid_batch_size] = pred1
        start_idx1 = start_idx1 + fid_batch_size
    # 特征空间是2048维空间，mu表示某一组数据在2048维特征空间上的均值（2048维）
    # 均值偏的越大，两组数据越不相似
    mu1 = np.mean(pred_arr1, axis=0) # np.ndarray([2048])
    # 协方差矩阵，表示高维数据的每一维上的方差（对角线元素）和不同维度的相关性（其他元素）
    # 简单衡量了分布的形状
    sigma1 = np.cov(pred_arr1, rowvar=False) # np.ndarray([2048, 2048])

    ''' 对另一组图像做同样的操作 '''
    files2 = [os.path.join(path2, fn) for fn in os.listdir(path2)]
    dataset2 = ImagePathDataset(files2, transforms=TF.ToTensor())
    dataloader2 = torch.utils.data.DataLoader(dataset2,
                                             batch_size=fid_batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=1)
    pred_arr2 = np.empty((len(files2), dims))
    start_idx2 = 0
    for batch2 in tqdm(dataloader2):
        batch2 = batch2.to(fid_device)
        with torch.no_grad():
            pred2 = model(batch2)[0]
        pred2 = pred2.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr2[start_idx2:start_idx2 + fid_batch_size] = pred2
        start_idx2 = start_idx2 + fid_batch_size
    mu2 = np.mean(pred_arr2, axis=0)
    sigma2 = np.cov(pred_arr2, rowvar=False)

    eps = 1e-6 # 数值稳定性
    diff = mu1 - mu2 # 均值差
    covmean = linalg.sqrtm(sigma1.dot(sigma2)) # 两个协方差矩阵相乘的整体平方根（不是逐元素平方根）
    tr_covmean = np.trace(covmean) # 迹
    fid_value = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    return fid_value