#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from torch.nn import Module, AvgPool2d, Linear
from models.base_module import Conv_Block, InvertedResidual


class PFLD(Module):
    def __init__(self, width_factor=1, input_size=112, landmark_number=98):
        '''
        __init__ 的 Docstring
        
        :param self: 实例指针
        :param width_factor: 宽度因子，用于调整网络宽度（即feature map的 channels 数量），论文中给出了0.25的情况
        :param input_size: 输入图像的尺寸（假设为正方形，边长）
        :param landmark_number: 需要预测的人脸关键点数量
        '''
        super(PFLD, self).__init__()
        # feature map: 112x112x3
        self.conv1 = Conv_Block(3, int(64 * width_factor), 3, 2, 1)
        # feature map: 56x56x(64*width_factor)
        self.conv2 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1, group=int(64 * width_factor))
        # feature map: 56x56x(64*width_factor), conv2 is DSC
        self.conv3_1 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 2, False, 2)
        self.conv3_2 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)
        self.conv3_3 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)
        self.conv3_4 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)
        self.conv3_5 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)
        # feature map: 28x28x(64*width_factor)
        self.conv4 = InvertedResidual(int(64 * width_factor), int(128 * width_factor), 2, False, 2)
        # feature map: 14x14x(128*width_factor)
        # 这里抽头给 Auxiliary Network 使用？
        self.conv5_1 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, False, 4)
        self.conv5_2 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_3 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_4 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_5 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_6 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        # feature map: 14x14x(128*width_factor)
        self.conv6 = InvertedResidual(int(128 * width_factor), int(16 * width_factor), 1, False, 2)
        # feature map: 14x14x(16*width_factor) --> S1
        self.conv7 = Conv_Block(int(16 * width_factor), int(32 * width_factor), 3, 2, 1)
        # feature map: 7x7x(32*width_factor) --> S2
        self.conv8 = Conv_Block(int(32 * width_factor), int(128 * width_factor), input_size // 16, 1, 0, has_bn=False)
        # feature map: 1x1x(128*width_factor) --> S3

        self.avg_pool1 = AvgPool2d(input_size // 8)
        # 池化核尺寸：14×14，用于S1特征图
        self.avg_pool2 = AvgPool2d(input_size // 16)
        # 池化核尺寸：7×7，用于S2特征图
        self.fc = Linear(int(176 * width_factor), landmark_number * 2)
        # 输入通道数：128*width_factor + 32*width_factor + 16*width_factor = 176*width_factor
        # 输出通道数：landmark_number * 2 （x,y坐标，有两个）
        # 第一个参数：输入维度
        # 第二个参数：输出维度

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)

        x = self.conv4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        x = self.conv5_6(x)

        # S1
        x = self.conv6(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        # S2
        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)
        
        # S3
        x3 = self.conv8(x)
        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks
