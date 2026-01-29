#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


def Conv_Block(in_channel, out_channel, kernel_size, stride, padding, group=1, has_bn=True, is_linear=False):
    '''
    Conv_Block 的 Docstring
    
    :param in_channel: 输入通道数
    :param out_channel: 输出通道数
    :param kernel_size: 卷积核尺寸，类型是 int 或则和 tuple
    :param stride: 卷积的步长，类型是 int
    :param padding: 边缘填充的大小，用于确保 feature map 尺寸不会缩小，与 stride 配合使用，类型是 int
    :param group: groups = group，默认为1，表示标准卷积；groups = in_channel = out_channel，则表示深度可分离卷积
    :param has_bn: 是否使用批归一化，类型是 bool
    :param is_linear: 是否使用线性激活函数，类型是 bool
    '''

    return Sequential(
        Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding, groups=group, bias=False),
        BatchNorm2d(out_channel) if has_bn else Sequential(),
        ReLU(inplace=True) if not is_linear else Sequential()
    )


class InvertedResidual(Module):
    def __init__(self, in_channel, out_channel, stride, use_res_connect, expand_ratio):
        super(InvertedResidual, self).__init__()
        # 调用父类构造函数，这样才能使用.cuda(), .parameters(), .state_dict()等方法以及forward魔术方法
        self.stride = stride
        assert stride in [1, 2]
        # 含义: 这是一个断言（Assertion）语句。它检查传入的 stride 值是否属于列表 [1, 2] 中的一个。
        # 作用: 参数校验，确保传入的 stride 是有效的，防止后续计算出错。

        exp_channel = in_channel * expand_ratio
        self.use_res_connect = use_res_connect
        self.inv_res = Sequential(
            Conv_Block(in_channel=in_channel, out_channel=exp_channel, kernel_size=1, stride=1, padding=0),
            Conv_Block(in_channel=exp_channel, out_channel=exp_channel, kernel_size=3, stride=stride, padding=1,
                       group=exp_channel),
            Conv_Block(in_channel=exp_channel, out_channel=out_channel, kernel_size=1, stride=1, padding=0,
                       is_linear=True)
            # 在将高维特征压缩回低维（Bottleneck）时，如果使用 ReLU 这样的非线性激活函数，会破坏由于维度压缩而仅存的特征信息。因此，在最后一层去掉了激活函数，使用线性输出。
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_res(x)
        # 条件：在 MobileNetV2 的标准定义中，只有当 Stride=1 且 输入通道数等于输出通道数 时，才使用残差连接。这个连接帮助梯度在深层网络中更好地传播，避免梯度消失。
        else:
            return self.inv_res(x)


class GhostModule(Module):
    '''
    内部操作：标准卷积采用Pointwise Conv, 剩余的 feature maps 通过 Depthwise Conv 生成
    '''
    def __init__(self, in_channel, out_channel, is_linear=False):
        super(GhostModule, self).__init__()
        self.out_channel = out_channel
        init_channel = math.ceil(out_channel / 2)
        # 先利用标准卷积生成一半数量的 feature maps
        new_channel = init_channel
        # 需要利用用标准卷积生成的 feature maps 通过廉价操作（如深度卷积）生成剩余的 feature maps

        self.primary_conv = Conv_Block(in_channel, init_channel, 1, 1, 0, is_linear=is_linear)
        self.cheap_operation = Conv_Block(init_channel, new_channel, 3, 1, 1, group=init_channel, is_linear=is_linear)
        # group=init_channel 表示这是一个深度卷积（Depthwise Convolution），即廉价操作

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channel, :, :]
        # 切片的意思是保留前面的 out_channel 个通道，丢弃多余的通道，因为在out_channel为奇数时会多生成一个通道
        # PyTorch 的 Tensor 维度： N, C, H, W


class GhostBottleneck(Module):
    def __init__(self, in_channel, hidden_channel, out_channel, stride):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.ghost_conv = Sequential(
            # GhostModule, 升维(Channels)
            GhostModule(in_channel, hidden_channel, is_linear=False),
            # DepthwiseConv-linear, 3×3 DWConv
            # stride = 1时，不做这一层 DWConv
            Conv_Block(hidden_channel, hidden_channel, 3, stride, 1, group=hidden_channel,
                       is_linear=True) if stride == 2 else Sequential(),
            # GhostModule-linear, 降维(Channels)，最后一层不做 ReLU 激活
            GhostModule(hidden_channel, out_channel, is_linear=True)
        )

        if stride == 1 and in_channel == out_channel:
            self.shortcut = Sequential()
        else:
            # shortcut 不可以直接相加时，使用 DWConv + PWConv 来调整维度和尺寸
            self.shortcut = Sequential(
                Conv_Block(in_channel, in_channel, 3, stride, 1, group=in_channel, is_linear=True),
                Conv_Block(in_channel, out_channel, 1, 1, 0, is_linear=True)
            )

    def forward(self, x):
        return self.ghost_conv(x) + self.shortcut(x)


class GhostOneModule(Module):
    def __init__(self, in_channel, out_channel, is_linear=False, inference_mode=False, num_conv_branches=1):
        super(GhostOneModule, self).__init__()
        self.out_channel = out_channel
        half_outchannel = math.ceil(out_channel / 2)

        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches

        # 主卷积（升维/特征变换）：使用 MobileOneBlock 替代原有的 Pointwise Conv (1x1)
        # 训练时多分支提取丰富特征，推理时重参数化为单层卷积
        self.primary_conv = MobileOneBlock(in_channels=in_channel,
                                           out_channels=half_outchannel,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           groups=1, # important
                                           inference_mode=self.inference_mode, # important
                                           use_se=False,
                                           num_conv_branches=self.num_conv_branches,
                                           is_linear=is_linear)
        
        # 廉价操作（特征生成）：使用 MobileOneBlock 替代原有的 Depthwise Conv (3x3)
        # 同样利用多分支结构增强特征提取能力
        self.cheap_operation = MobileOneBlock(in_channels=half_outchannel,
                                              out_channels=half_outchannel,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              groups=half_outchannel, # important 
                                              inference_mode=self.inference_mode, # important
                                              use_se=False,
                                              num_conv_branches=self.num_conv_branches,
                                              is_linear=is_linear)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # 拼接两个分支的特征图，形成完整的输出
        # 注意：GhostOneModule 直接拼接，没有像原始 GhostModule 那样做切片，
        # 这意味着 out_channel 最好是偶数，否则 math.ceil 会导致实际输出通道比 out_channel 多 1
        out = torch.cat([x1, x2], dim=1)
        return out


class GhostOneBottleneck(Module):
    def __init__(self, in_channel, hidden_channel, out_channel, stride, inference_mode=False, num_conv_branches=1):
        super(GhostOneBottleneck, self).__init__()
        assert stride in [1, 2]

        self.inference_mode = inference_mode
        self.num_conv_branches = num_conv_branches

        self.ghost_conv = Sequential(
            # GhostModule (升维): 使用 MobileOne 增强版的 Ghost 模块
            GhostOneModule(in_channel, hidden_channel, is_linear=False, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches),
            
            # Depthwise Conv (提取空间特征): 
            # 注意：这里的实现逻辑是 stride=2 时才加 DWConv, stride=1 时中间没有 DWConv。
            # 这意味着当 stride=1 时，整个 Bottleneck 只有两个 1x1 变换 (GhostOneModule)，这比较特殊。
            MobileOneBlock(in_channels=hidden_channel,
                           out_channels=hidden_channel,
                           kernel_size=3,
                           stride=stride,
                           padding=1,
                           groups=hidden_channel,
                           inference_mode=self.inference_mode,
                           use_se=False,
                           num_conv_branches=self.num_conv_branches,
                           is_linear=True) if stride == 2 else Sequential(),
            
            # GhostModule (降维): 投影回低维空间，无激活函数 (is_linear=True)
            GhostOneModule(hidden_channel, out_channel, is_linear=True, inference_mode=self.inference_mode, num_conv_branches=self.num_conv_branches)
        )
        
        # 注意：这里移除了 Shortcut (Residual Connection)
        # 原始 GhostBottleneck 在 stride=1 时会有 x + ghost_conv(x)
        # 这里参考 YOLOv7 / MobileOne 的发现：当使用重参数化模块时，简单的残差相加可能会破坏重参数化带来的增益
        # 因此 GhostOneBottleneck 是一个直通结构 (Straight-through structure)

    def forward(self, x):
        return self.ghost_conv(x)


class SEBlock(nn.Module):
    """ Squeeze and Excite module.

        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    """ MobileOne building block.

        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1,
                 is_linear: bool = False) -> None:
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches 
        # num_conv_branches 卷积分支个数

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if is_linear:
            self.activation = nn.Identity()
        else:
            self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            # 定义一个可重参数化的跳跃连接分支（Skip Connection）
            # 仅当输入输出通道数相同且步长为1时存在，由一个 BatchNorm 层构成
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches
            # 定义可重参数化的卷积分支列表
            # 这是一个包含 num_conv_branches 个分支的模块列表，每个分支都是一个 standard conv + bn
            # 这种过参数化（Over-parameterization）策略有助于在训练阶段丰富特征空间，提升模型性能
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            # 定义可重参数化的缩放分支（Scale Branch）
            # 这是一个 kernel_size=1 的卷积分支，用于捕捉像素级的特征变换
            # 仅当主卷积核大于 1x1 时才使用，以提供不同感受野的特征补充
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        # 推理模式下的前向传播：直接使用重参数化后的单层卷积进行计算
        # 此时结构已等效于一个标准的 Conv-BN-Act 模块，速度极快
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # 训练模式下的多分支前向传播：将所有分支的输出相加
        
        # Skip branch output (跳跃连接分支输出)
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output (1x1 缩放分支输出)
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches (主卷积分支输出)
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        # 最终经过 SE 模块（可选）和激活函数
        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        # 重参数化核心方法：将训练时的多分支结构融合为一个等效的单层卷积
        if self.inference_mode:
            return
        
        # 获取融合后的权重 kernel 和偏置 bias
        # _get_kernel_bias() 会将 rbr_skip, rbr_conv, rbr_scale 三个分支的参数进行合并
        kernel, bias = self._get_kernel_bias()
        
        # 创建一个新的单层卷积 (reparam_conv) 来替代原来的多分支结构
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        # 将融合后的参数赋值给新卷积层
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        # 删除不再需要的分支以释放内存，并标记模型为推理模式 (inference_mode = True)
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list
