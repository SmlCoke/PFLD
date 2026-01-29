#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import torch
from torch.nn import Module, AvgPool2d, Linear
from models.base_module import MobileOneBlock, GhostOneBottleneck, Conv_Block


class PFLD_GhostOne(Module):
    def __init__(self, width_factor=1, input_size=112, landmark_number=98, inference_mode=False):
        """
        PFLD_GhostOne 构造函数

        :param width_factor: 宽度因子，用于控制网络通道数的缩放比例
        :param input_size: 输入图像的尺寸，默认为 112x112
        :param landmark_number: 关键点数量，默认为 98
        :param inference_mode: 是否为推理模式。
                               如果为 True，MobileOneBlock 将会被重参数化为单层卷积，加速推理。
                               如果为 False，则保持多分支结构用于训练。
        """
        super(PFLD_GhostOne, self).__init__()

        self.inference_mode = inference_mode
        self.num_conv_branches = 6 # MobileOneBlock 中的卷积分支数量，用于增加训练时的参数量

        # 这里的 conv1 和 conv2 对应原始 ResNet/MobileNet 的 stem 部分
        # 使用 MobileOneBlock 替代标准卷积，能够在不增加推理耗时的情况下提升性能

        # Conv3x3, stride=2, padding=1 -> 下采样
        self.conv1 = MobileOneBlock(in_channels=3,
                                    out_channels=int(64 * width_factor),
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    groups=1,
                                    inference_mode=self.inference_mode,
                                    use_se=False,
                                    num_conv_branches=self.num_conv_branches,
                                    is_linear=False)
        # Depthwise Conv3x3 (groups=in_channels)
        self.conv2 = MobileOneBlock(in_channels=int(64 * width_factor),
                                    out_channels=int(64 * width_factor),
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=int(64 * width_factor),
                                    inference_mode=self.inference_mode,
                                    use_se=False,
                                    num_conv_branches=self.num_conv_branches,
                                    is_linear=False)

        # Bottleneck 堆叠部分
        # 核心变革：使用 GhostOneBottleneck (基于 MobileOne 的 Ghost 模块)
        # 结构：GhostOneModule (升维) -> DWConv (可选下采样) -> GhostOneModule (降维)

        # Stage 1: Stride=2 下采样, 输出通道 64 -> 80
        self.conv3_1 = GhostOneBottleneck(int(64 * width_factor), int(96 * width_factor), int(80 * width_factor), stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv3_2 = GhostOneBottleneck(int(80 * width_factor), int(120 * width_factor), int(80 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv3_3 = GhostOneBottleneck(int(80 * width_factor), int(120 * width_factor), int(80 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)

        # Stage 2: Stride=2 下采样, 输出通道 80 -> 96
        self.conv4_1 = GhostOneBottleneck(int(80 * width_factor), int(200 * width_factor), int(96 * width_factor), stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv4_2 = GhostOneBottleneck(int(96 * width_factor), int(240 * width_factor), int(96 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv4_3 = GhostOneBottleneck(int(96 * width_factor), int(240 * width_factor), int(96 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)

        # Stage 3: Stride=2 下采样, 输出通道 96 -> 144
        self.conv5_1 = GhostOneBottleneck(int(96 * width_factor), int(336 * width_factor), int(144 * width_factor), stride=2, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv5_2 = GhostOneBottleneck(int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv5_3 = GhostOneBottleneck(int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv5_4 = GhostOneBottleneck(int(144 * width_factor), int(504 * width_factor), int(144 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)

        # Final Layers
        self.conv6 = GhostOneBottleneck(int(144 * width_factor), int(216 * width_factor), int(16 * width_factor), stride=1, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        self.conv7 = MobileOneBlock(in_channels=int(16 * width_factor),
                                    out_channels=int(32 * width_factor),
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=1,
                                    inference_mode=self.inference_mode,
                                    use_se=False,
                                    num_conv_branches=self.num_conv_branches,
                                    is_linear=False)
        self.conv8 = Conv_Block(int(32 * width_factor), int(128 * width_factor), input_size // 16, 1, 0, has_bn=False)

        # 多尺度特征融合所需的池化层
        self.avg_pool1 = AvgPool2d(input_size // 2)
        self.avg_pool2 = AvgPool2d(input_size // 4)
        self.avg_pool3 = AvgPool2d(input_size // 8)
        self.avg_pool4 = AvgPool2d(input_size // 16)

        # 全连接层：输入维度是各个尺度特征展平后的总和
        self.fc = Linear(int(512 * width_factor), landmark_number * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # S1 (Stride=2): 提取第一层特征并池化
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        # S2 (Stride=4): 提取第二层特征并池化
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        # S3 (Stride=8): 提取第三层特征并池化
        x3 = self.avg_pool3(x)
        x3 = x3.view(x3.size(0), -1)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        # S4 (Stride=16): 提取第四层特征并池化
        x4 = self.avg_pool4(x)
        x4 = x4.view(x4.size(0), -1)

        x = self.conv6(x)
        x = self.conv7(x)
        x5 = self.conv8(x)
        x5 = x5.view(x5.size(0), -1)

        # MSFF (Multi-Scale Feature Fusion): 在通道维度拼接所有尺度的特征
        multi_scale = torch.cat([x1, x2, x3, x4, x5], 1)
        landmarks = self.fc(multi_scale)

        return landmarks
