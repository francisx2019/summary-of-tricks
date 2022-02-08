# -*- coding:utf-8 -*-
"""
Atrous spatial pyramid pooling
论文：https://arxiv.org/pdf/1606.00915.pdf
带有空洞卷积的空间金字塔池化模块，主要是为了提高网络的感受野，并引入多尺度信息而提出来的．
1. 全局池化 + １x１卷积 + 双线性插值
2. 三个３x３的空洞卷积，１个１x１的卷积
3. 上述五个部分在channel维度进行concat在送入１x1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1_1 = nn.Conv2d(in_channel, depth, (1, 1), (1, 1))
        self.atrous_block1 = nn.Conv2d(in_channel, depth, (1, 1), (1, 1))
        self.atrous_block6 = nn.Conv2d(in_channel, depth, (3, 3), (1, 1),
                                       padding=(6, 6), dilation=(6, 6))
        self.atrous_block12 = nn.Conv2d(in_channel, depth, (3, 3), (1, 1),
                                        padding=(12, 12), dilation=(12, 12))
        self.atrous_block18 = nn.Conv2d(in_channel, depth, (3, 3), (1, 1),
                                        padding=(18, 18), dilation=(18, 18))
        self.conv1_1_out = nn.Conv2d(depth * 5, depth, (1, 1), (1, 1))

    def forward(self, x):
        size = x.shape[2:]
        # part 1
        feat = self.mean(x)
        feat = self.conv1_1(feat)
        feat = F.interpolate(feat, size=(int(size[0]), int(size[1])),
                             mode='bilinear', align_corners=True)

        # part 2
        atrous_1 = self.atrous_block1(x)
        atrous_6 = self.atrous_block6(x)
        atrous_12 = self.atrous_block12(x)
        atrous_18 = self.atrous_block18(x)

        # part 3
        print(feat.size(), atrous_1.size(), atrous_6.size(), atrous_12.size(), atrous_18.size())
        out = torch.cat([feat, atrous_1, atrous_6, atrous_12, atrous_18], dim=1)
        out = self.conv1_1_out(out)
        return out


if __name__ == "__main__":
    x = torch.randn((2, 512, 128, 128))
    aspp = ASPP()
    aspp(x)