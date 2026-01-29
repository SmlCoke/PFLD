#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import torch
from torch.nn import Module, AvgPool2d, Linear
from models.base_module import Conv_Block, GhostBottleneck
import torch.nn.functional as F


class PFLD_GhostNet(Module):
    def __init__(self, width_factor=1, input_size=112, landmark_number=98):
        """
        PFLD_GhostNet 构造函数
        该模型将 PFLD 的主干网络替换为 GhostNet，利用 GhostModule 进行轻量化特征提取。

        :param width_factor: 宽度因子，用于控制网络通道数的缩放比例
        :param input_size: 输入图像的尺寸，默认为 112x112
        :param landmark_number: 关键点数量，默认为 98
        """
        super(PFLD_GhostNet, self).__init__()

        # Conv1: 标准卷积 (3 -> 64), stride=2 (下采样)
        self.conv1 = Conv_Block(3, int(64 * width_factor), 3, 2, 1)
        
        # Conv2: 标准卷积 (dw/groups=in_channels), stride=1
        self.conv2 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1, group=int(64 * width_factor))

        # GhostBottleneck 堆叠部分
        # 核心变革：使用 GhostBottleneck (基于 GhostNet 的 Ghost 模块)
        # 结构：GhostModule (升维) -> DWConv (可选下采样) -> GhostModule (降维)

        # Stage 1: stride=2 下采样, 通道 64 -> 80
        # 由一个 Stride=2 的 bottleneck 和 两个 Stride=1 的 bottleneck 组成
        self.conv3_1 = GhostBottleneck(int(64 * width_factor), int(128 * width_factor), int(80 * width_factor), stride=2)
        self.conv3_2 = GhostBottleneck(int(80 * width_factor), int(160 * width_factor), int(80 * width_factor), stride=1)
        self.conv3_3 = GhostBottleneck(int(80 * width_factor), int(160 * width_factor), int(80 * width_factor), stride=1)

        # Stage 2: stride=2 下采样, 通道 80 -> 96
        # 由一个 Stride=2 的 bottleneck 和 两个 Stride=1 的 bottleneck 组成
        self.conv4_1 = GhostBottleneck(int(80 * width_factor), int(240 * width_factor), int(96 * width_factor), stride=2)
        self.conv4_2 = GhostBottleneck(int(96 * width_factor), int(288 * width_factor), int(96 * width_factor), stride=1)
        self.conv4_3 = GhostBottleneck(int(96 * width_factor), int(288 * width_factor), int(96 * width_factor), stride=1)

        # Stage 3: stride=2 下采样, 通道 96 -> 144
        # 由一个 Stride=2 的 bottleneck 和 三个 Stride=1 的 bottleneck 组成
        self.conv5_1 = GhostBottleneck(int(96 * width_factor), int(384 * width_factor), int(144 * width_factor), stride=2)
        self.conv5_2 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)
        self.conv5_3 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)
        self.conv5_4 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)

        # Final Layers: 进一步处理特征，降低参数量
        # 通道 144 -> 16
        self.conv6 = GhostBottleneck(int(144 * width_factor), int(288 * width_factor), int(16 * width_factor), stride=1)
        # 标准卷积 (3x3), 通道 16 -> 32
        self.conv7 = Conv_Block(int(16 * width_factor), int(32 * width_factor), 3, 1, 1)
        # 最终卷积 block，无 BN (通道 32 -> 128)
        self.conv8 = Conv_Block(int(32 * width_factor), int(128 * width_factor), input_size // 16, 1, 0, has_bn=False)

        # 多尺度特征池化层 (MSFF 准备工作)
        self.avg_pool1 = AvgPool2d(input_size // 2)
        self.avg_pool2 = AvgPool2d(input_size // 4)
        self.avg_pool3 = AvgPool2d(input_size // 8)
        self.avg_pool4 = AvgPool2d(input_size // 16)

        # 线性全连接层
        self.fc = Linear(int(512 * width_factor), landmark_number * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # S1 特征
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        # S2 特征
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        # S3 特征
        x3 = self.avg_pool3(x)
        x3 = x3.view(x3.size(0), -1)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        # S4 特征
        x4 = self.avg_pool4(x)
        x4 = x4.view(x4.size(0), -1)

        x = self.conv6(x)
        x = self.conv7(x)
        x5 = self.conv8(x)
        x5 = x5.view(x5.size(0), -1)

        # 多尺度融合
        multi_scale = torch.cat([x1, x2, x3, x4, x5], 1)
        landmarks = self.fc(multi_scale)

        return landmarks

# 辅助网络，但是不像原始网络一样回归欧拉角
# 数据类型：最终输出的是一个 4维张量 (Tensor)，形状为 (Batch_Size, 1, H, W)。其中 H 和 W 是 out1 的高和宽（通常比较大，接近原图的 1/2 或 1/4）。
# 回归内容：它不是回归坐标点，也不是回归角度 (Pitch, Yaw, Roll)。
# 因为它输出的是一张单通道的图，这通常是一个 二值掩码 (Binary Mask) 或者 热力图 (Heatmap/Attention Map)。
# 在很多改进版的 PFLD 任务中，这种结构通常用来做 人脸分割 (Segmentation) 或者 前景/背景分类。
# 目的：它强制主干网络去学习“哪里是人脸，哪里是背景”。通过这个辅助任务，模型能更好地理解人脸的轮廓和结构，从而间接提高关键点预测的精度。
class PFLD_Ultralight_AuxiliaryNet(Module):
    def __init__(self, width_factor=1):
        super(PFLD_Ghost_AuxiliaryNet, self).__init__()
        # 特征对齐层，四个 1×1 卷积层分别接受来自 back bone 的四个不同尺度的特征图，压缩到相同的通道数后再进行融合
        self.conv1 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv2 = Conv_Block(int(80 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv3 = Conv_Block(int(96 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv4 = Conv_Block(int(144 * width_factor), int(64 * width_factor), 1, 1, 0)
        
        # 用作特征平滑层
        self.merge1 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)
        self.merge2 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)
        self.merge3 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)

        # 输出层，将通道直接压为1，最终 feature maps 的形状为 [B, 1, H, W]
        self.conv_out = Conv_Block(int(64 * width_factor), 1, 1, 1, 0)

    def forward(self, out1, out2, out3, out4):
        # 特征对齐层，四个 1×1 卷积层分别接受来自 back bone 的四个不同尺度的特征图，压缩到相同的通道数后再进行融合
        output1 = self.conv1(out1)
        output2 = self.conv2(out2)
        output3 = self.conv3(out3)
        output4 = self.conv4(out4)

        # 上采样层 + 特征融合层 + 特征平滑层：
        # 上采样层：三个F.interpolate 将深层、小尺寸的特征图（如 output4）放大，使其尺寸与浅层、大尺寸的特征图（如 output3）一致。
        # F.interpolate：插值/采样函数，nearst为最近邻插值，还有bilinear双线性插值等
        up4 = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="nearest")
       # 特征融合：将“高语义信息”（来自深层）与“高分辨率空间信息”（来自浅层）进行逐元素相加。这是 FPN 的核心思想，能让特征图既看得清细节，又懂整体语义。
        output3 = output3 + up4
        # 特征平滑：在融合之后，接了一个 3x3 的卷积。消除上采样带来的混叠效应（Aliasing Effect），平滑特征图，融合不同来源的特征。
        output3 = self.merge3(output3)

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        output1 = self.conv_out(output1)

        return output1
