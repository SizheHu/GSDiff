import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        # 防止反向传播更新
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class BoundaryModel(nn.Module):
    # 输入是3个通道的RGB图，输出还是相同通道数
    def __init__(self):
        super(BoundaryModel, self).__init__()

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.shortcut2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.shortcut3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.shortcut4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.shortcut5 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )


        self.Up5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.Up_conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.shortcut_Up_conv5 = Identity()
        

        self.Up4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.Up_conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.shortcut_Up_conv4 = Identity()

        self.Up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.Up_conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.shortcut_Up_conv3 = Identity()

        self.Up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.Up_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            FrozenBatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.shortcut_Up_conv2 = Identity()

        self.Conv = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # print('x', x.shape) torch.Size([16, 3, 256, 256])
        e1 = self.Conv1(x)
        # print('e1', e1.shape) torch.Size([16, 64, 256, 256])
        
        e2 = self.Maxpool1(e1)
        # print('e2', e2.shape) torch.Size([16, 64, 128, 128])

        e2 = self.Conv2(e2) + self.shortcut2(e2)
        # print('e2', e2.shape) torch.Size([16, 128, 128, 128])

        e3 = self.Maxpool2(e2)
        # print('e3', e3.shape) torch.Size([16, 128, 64, 64])
        e3 = self.Conv3(e3) + self.shortcut3(e3)
        # print('e3', e3.shape) torch.Size([16, 256, 64, 64]) # 

        e4 = self.Maxpool3(e3)
        # print('e4', e4.shape) torch.Size([16, 256, 32, 32])
        e4 = self.Conv4(e4) + self.shortcut4(e4)
        # print('e4', e4.shape) torch.Size([16, 512, 32, 32])

        e5 = self.Maxpool4(e4)
        # print('e5', e5.shape) torch.Size([16, 512, 16, 16])
        e5 = self.Conv5(e5) + self.shortcut5(e5)
        # print('e5', e5.shape) torch.Size([16, 1024, 16, 16])

        d5 = self.Up5(e5)
        # print('d5', d5.shape) torch.Size([16, 512, 32, 32])

        d5 = self.Up_conv5(d5) + self.shortcut_Up_conv5(d5)
        # print('d5', d5.shape) torch.Size([16, 512, 32, 32])

        d4 = self.Up4(d5)
        # print('d4', d4.shape) torch.Size([16, 256, 64, 64])

        d4 = self.Up_conv4(d4) + self.shortcut_Up_conv4(d4)
        # print('d4', d4.shape) torch.Size([16, 256, 64, 64])

        d3 = self.Up3(d4)
        # print('d3', d3.shape) torch.Size([16, 128, 128, 128])

        d3 = self.Up_conv3(d3) + self.shortcut_Up_conv3(d3)
        # print('d3', d3.shape) torch.Size([16, 128, 128, 128])

        d2 = self.Up2(d3)
        # print('d2', d2.shape) torch.Size([16, 64, 256, 256])

        d2 = self.Up_conv2(d2) + self.shortcut_Up_conv2(d2)
        # print('d2', d2.shape) torch.Size([16, 64, 256, 256])

        out = self.Conv(d2)
        # print('out', out.shape) torch.Size([16, 3, 256, 256])


        return out