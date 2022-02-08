# -*- coding:utf-8 -*-
"""
Squeeze-and-Excitation Networks
论文：https://arxiv.org/pdf/1709.01507.pdf
一种通道注意力机制，可以自适应的调整各通道的特征响应值．
"""
import torch
import torch.nn as nn


class SE(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            *[nn.Linear(in_channel, in_channel // reduction, bias=False),
              nn.ReLU(inplace=True),
              nn.Linear(in_channel // reduction, in_channel, bias=False),
              nn.Sigmoid()])

    def forward(self, x):
        b, c, _, _ = x.size()
        # squeeze 在 H 和 W 通道
        y = self.avg_pool(x).view(int(b), int(c))
        print(y.size())
        y = self.fc(y).view(int(b), int(c), 1, 1)
        print(y.size())
        print(y.expand_as(x))
        return x * y.expand_as(x)


if __name__ == "__main__":
    x = torch.randn((2, 16, 224, 224))
    se = SE(16)
    se(x)
