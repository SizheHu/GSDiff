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

def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)

def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=378,
                            ret_var=True, output=sys.stdout, **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
        for i in bar:
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)

def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est




def kid_boundary_smaller_subsetsize(path1, path2, kid_batch_size, kid_device):
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
    pred_arr1 = np.empty((len(files1), dims)) # np.ndarray([len(test set), 2048])
    start_idx1 = 0
    for batch1 in tqdm(dataloader1):
        batch1 = batch1.to(kid_device) # torch.Size([64, 3, 256, 256])
        '''FID的计算器中，我们也是用了inception网络。
        inception其实就是特征提取的网络，最后一层输出图像的类别。
        不过我们会去除最后的全连接或者池化层，使得我们得到一个2048维度的特征。'''
        with torch.no_grad():
            pred1 = model(batch1)[0] # torch.Size([64, 2048, 1, 1])
        pred1 = pred1.squeeze(3).squeeze(2).cpu().numpy() # np.ndarray([64, 2048])
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
    kid_values = polynomial_mmd_averages(pred_arr1, pred_arr2, n_subsets=100)

    return kid_values[0].mean() * 1000