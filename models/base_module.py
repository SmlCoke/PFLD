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
    """ MobileOne building block.（MobileOne 核心构建块）

        该模块采用了结构重参数化（Structural Re-parameterization）技术：
        - 训练时（Train-time）：采用多分支架构（Multi-branched architecture），通过增加并行的卷积分支来获得更强的特征提取能力和更平滑的损失地形。
        - 推理时（Inference-time）：将多分支融合为单路 CNN 架构（Plain-CNN style），大幅减少显存访问开销，提升推理速度。
        
        更多细节请参考：
        `An Improved One millisecond Mobile Backbone` - https://arxiv.org/pdf/2206.04040.pdf
        `RepVGG: Making VGG-style ConvNets Great Again` - https://arxiv.org/pdf/2101.03697.pdf
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
        """ 构造 MobileOneBlock 模块。

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小 (通常为 3 或 1)
        :param stride: 步长
        :param padding: 填充
        :param dilation: 膨胀系数
        :param groups: 分组数 (groups=in_channels 时为 Depthwise Conv)
        :param inference_mode: 是否直接实例化为推理模式（如果为 True，则不构建多分支）
        :param use_se: 是否使用 SE-ReLU 激活 (Squeeze-and-Excitation)
        :param num_conv_branches: 训练时使用的并行卷积分支数量 (默认为 1，即 RepVGG 风格)
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches 

        # 1. 激活函数与注意力机制
        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if is_linear:
            self.activation = nn.Identity()
        else:
            self.activation = nn.ReLU()

        # 2. 构建卷积层（训练模式 vs 推理模式）
        if inference_mode:
            # 推理模式：直接构建一个标准的单路 Conv2d，没有 BN，只有 Bias
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # 训练模式：构建多分支结构 (Over-parameterization)
            
            # 分支 A: Skip Connection (跳跃连接)
            # 仅当输入输出通道一致且 stride=1 时才存在
            # 注意：这里的 Skip 不是简单的相加，而是一个 BatchNorm 层，这也是可以被重参数化进卷积里的
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # 分支 B: Conv Branches (主卷积分支)
            # 包含 num_conv_branches 个并行的 (Conv + BN) 结构
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # 分支 C: Scale Branch (1x1 缩放分支)
            # 用 1x1 卷积来模拟全连接或通道变换，捕捉不同于 3x3 的特征
            # 仅当主卷积核 > 1 (如 3x3) 时才添加，否则如果是 1x1 卷积本身，这个分支就冗余了
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 前向传播逻辑 """
        
        # 模式一：推理模式
        # 此时多分支已被“吸收”进 self.reparam_conv 中
        # 计算路径：Input -> Conv(已融合) -> SE(可选) -> Activation -> Output
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # 模式二：训练模式 (多分支相加)
        # 计算路径：Input -> [Skip_BN + Scale_1x1 + Conv_NxN_0 + Conv_NxN_1...] -> Sum -> SE -> Act -> Output
        
        # 1. Skip branch 分支输出
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # 2. Scale branch (1x1) 分支输出
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # 3. Main Conv branches 分支输出累加
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        # 4. 经过 SE 模块和激活函数
        return self.activation(self.se(out))

    def reparameterize(self):
        """ 重参数化核心函数
        
        功能：将训练时的多分支（Conv_A + BN, Conv_B + BN, BN_Identity...）
             等效转换为推理时的单层卷积（Conv_Fused + Bias）。
             
        调用时机：在 export ONNX 之前，或者在 model.eval() 且不再训练之后手动调用。
        """
        if self.inference_mode:
            return
        
        # 1. 计算融合后的 Kernel 和 Bias
        # 这是数学推导的落地实现，将所有分支的参数合并
        kernel, bias = self._get_kernel_bias()
        
        # 2. 创建全新的单层卷积 (reparam_conv)
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True) 
        # 注意：bias参数默认不开启， 但此处必须开启 Bias，因为 BN 的 beta 参数会被融合进这里
        
        # 3. 填充权重
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # 4. 删除冗余分支，释放内存
        for para in self.parameters():
            para.detach_()
        # 它将张量从当前的计算图中剥离出来。剥离后，这些张量就不再需要计算梯度（requires_grad 会变为 False），也不再指向原来的父节点。
        # 确保旧权重变成纯粹的数据（Data），而不是计算图的一部分。防止在后续操作中意外触发自动求导逻辑报错。还可以切断引用链，帮助 Python 的垃圾回收器（GC）更快地识别出这些内存是可以回收的。
        self.__delattr__('rbr_conv')
        # 这一步是通过 Python 的魔术方法来删除属性。它等价于 del self.rbr_conv，但在 PyTorch 的 nn.Module 中，这个操作有特殊的含义。
        # 当你使用 self.rbr_conv = nn.ModuleList(...) 时，PyTorch 不仅仅是给对象添加了一个属性，它还把这个 rbr_conv 注册到了模块内部的一个字典 _modules 中。正是因为在这个字典里，当你调用 model.cuda() 或 model.to(device) 时，主模型才能顺藤摸瓜找到子模块并把它们也挪到 GPU 上。
        # __delattr__ 的作用：当你调用删除操作时，PyTorch 会拦截这个操作，做两件事：(1) 从 python 对象的属性字典 __dict__ 中删除 rbr_conv 变量名。(2) 关键点：从 PyTorch 内部的 _modules 注册表中注销该模块。这样做之后，MobileOneBlock 对象就不再持有对 rbr_conv 的引用，PyTorch 也不会再把它当作子模块来处理。
        
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        # 5. 标记为推理模式，改变 forward 的行为
        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 获取融合后的权重和偏置。
        核心思想：利用卷积操作的线性可加性 -> Conv(x, W1) + Conv(x, W2) = Conv(x, W1+W2)
        """
        # 1. 处理 Scale 分支 (1x1 Conv + BN)
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            # 先融合 Conv+BN -> 等效 W, b
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # _fuse_bn_tensor 返回的是 conv 和 bn 合并之后的 conv_weight 和 conv_bias
            # 因为 Scale 分支是 1x1 卷积，而主分支是 KxK (如 3x3)
            # 需要将 1x1 核周围补零 (Pad) 变成 3x3，才能直接与主卷积核相加
            pad = self.kernel_size // 2
            # 如果 MobileOneBlock 本身是 3×3 卷积，则 kernel_size=3，则 pad=1
            # 如果 MobileOneBlock 本身是 1×1 卷积，则 pad = 0
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])
            # [pad, pad, pad, pad] 分别对应 左右上下 四个方向的填充大小为 pad 行/列的像素

        # 2. 处理 Skip 分支 (Only BN)
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            # BN 层本质上是一个特殊的 1x1 卷积（属于恒等映射的变体）
            # _fuse_bn_tensor 能够将其转化为一个卷积核里大部分是 0，中间是 1 的特殊卷积核
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # 3. 处理 Main Conv 分支 (NxN Conv + BN)
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            # 直接累加权重
            kernel_conv += _kernel
            bias_conv += _bias

        # 4. 最终合并：所有分支的 Kernel 相加，所有 Bias 相加
        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 分支融合原子操作：将 (Conv+BN) 或 (BN) 融合为 (Conv_weight, Conv_bias)。
        
        原理：
        BN 公式: y = (x - mean) / sqrt(var + eps) * gamma + beta
        卷积公式: y = Wx 
        融合后: y = (W * gamma / std) * x + (beta - mean * gamma / std)
        
        :param branch: 输入分支，可能是 nn.Sequential(Conv, BN) 或者单独的 nn.BatchNorm2d
        """
        if isinstance(branch, nn.Sequential):
            # Case 1: 分支是 Conv + BN
            # 为什么是 .conv 和 .bn？详见本类的辅助函数 .conv_bn 的命名约定
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # Case 2: 分支只有 BN (Skip Connection)
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                # 构造一个恒等映射卷积核（Identity Kernel）
                # 这是一个 KxK 的卷积核，除了中心点是 1，其余都是 0
                input_dim = self.in_channels // self.groups
                # 这是考虑了 分组卷积 (Group Convolution) 的情况。
                # 如果是标准卷积，则self.groups = 1, 每个卷积核的深度等于输入通道数self.in_channels；如果是DWConv，则self.groups = self.in_channels, 每个卷积核的深度为1。
                kernel_value = torch.zeros((self.in_channels,
                                            # 按道理这里应该是self.out_channels，但是对应 Case 2: 分支只有 BN (Skip Connection)，这是恒等映射，self.in_channels == self.out_channels
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                # 构建了一个标准的 PyTorch 卷积权重容器，形状为：()[out_channels, in_channels/groups, K, K]，初始值全为0
                # 如果是标准卷积，则 input_dim = self.in_channels；对应标准卷积的卷积核需要处理所有通道；如果是 DWConv，则 input_dim = 1；对应 DWConv 的卷积核每个只处理一个通道。
                for i in range(self.in_channels):
                    # 按道理这里应该是self.out_channels，但是对应 Case 2: 分支只有 BN (Skip Connection)，这是恒等映射，self.in_channels == self.out_channels
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                    # 让卷积核中间的值为0
                    # 如果是标准卷积，input_dim = self.in_channels, 则 i % input_dim = i，即每隔通道的[self.kernel_size // 2, self.kernel_size // 2]位置为1
                    # 如果是 DWConv，input_dim = 1, 则 i % input_dim = 0，即只有每个卷积核的第0个输入通道的[self.kernel_size // 2, self.kernel_size // 2]位置为1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            
        # 融合公式实现
        std = (running_var + eps).sqrt()
        # t = gamma / std
        t = (gamma / std).reshape(-1, 1, 1, 1) # reshape 以支持广播乘法
        
        # 新 Weight = 旧 Weight * (gamma / std)
        # 新 Bias = beta - mean * (gamma / std)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ 辅助函数：快速创建一个 Conv2d + BatchNorm2d 组合。
            注意 Conv2d 的 bias 被设为 False，因为后面紧跟 BN 层，Bias 会被 BN 的 beta 参数抵消掉，没有存在的意义。
        """
        mod_list = nn.Sequential()
        # 为 Conv 层命名为 'conv'，方便后续 getattr 访问
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False)) 
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list
