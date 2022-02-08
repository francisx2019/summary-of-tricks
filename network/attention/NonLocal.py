# -*- coding:utf-8 -*-
"""
Non-local Neural Networks
论文：https://arxiv.org/pdf/1711.07971.pdf
让网络的感受野可以很大
"""
import torch
import torch.nn as nn


class NonLocal(nn.Module):
    def __init__(self, channel):
        super(NonLocal, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(channel, self.inter_channel,
                                  (1, 1), (1, 1), (0, 0), bias=False)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel,
                                    (1, 1), (1, 1), (0, 0), bias=False)
        self.conv_g = nn.Conv2d(channel, self.inter_channel,
                                (1, 1), (1, 1), (0, 0), bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel,
                                   (1, 1), (1, 1), (0, 0), bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        # [N, C, H, W] -> [N, C/2, H*W]
        x_phi = self.conv_phi(x).view(int(b), int(c), -1)
        # [N, C, H, W] -> [N, H*W, C/2]
        x_theta = self.conv_theta(x).view(int(b), int(c), -1).permute(0, 2, 1).contiguous()
        # [N, C, H, W] -> [N, H*W, C/2]
        x_g = self.conv_g(x).view(int(b), int(c), -1).permute(0, 2, 1).contiguous()
        # phi和theta进行矩阵乘
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)

        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).\
            contiguous().view(int(b), self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        return mask + x
