# -*- coding:utf-8 -*-
"""
Spatial Transformer Networks
论文：https://arxiv.org/pdf/1506.02025.pdf
该方法主要用在OCR任务中，STN模块将空间变换移植到网络中，提高
网络对旋转，平移，尺度等不变性．该模块能够把原始图像转换成网络
想要的图像，且过程是无监督的方式，变换参数是自动学习获取．
一般用在网络的前面层
"""
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        # 根据实际情况修改设计参数
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(7, 7)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=(5, 5)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Repressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        size = list([-1, int(x.size[1]), int(x.size[2]), int(x.size[3])])
        grid = F.affine_grid(theta, size)
        x = F.grid_sample(x, grid)
        return x
