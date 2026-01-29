import torch
from torch import nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, 
                attribute_gt, 
                landmark_gt, 
                euler_angle_gt, 
                angle, 
                landmarks, 
                train_batchsize):
        '''
        forward 的 Docstring
        
        :param self: 说明
        :param attribute_gt: 类型 torch.Tensor, 形状 (batch_size, n_attributes)包含每个样本属性的张量，例如性别、年龄
        :param landmark_gt: 类型 torch.Tensor, 形状 (batch_size, n_landmarks * 2 )包含每个样本地标点的张量，例如(x1, y1, x2, y2, ..., xN, yN)
        :param euler_angle_gt: 类型 torch.Tensor, 形状 (batch_size, 3)包含每个样本欧拉角的张量（俯仰角、偏航）角、滚转角
        :param angle: 类型 torch.Tensor, 形状 (batch_size, 3)包含预测的欧拉角的张量
        :param landmarks: 类型 torch.Tensor, 形状 (batch_size, n_landmarks * 2)包含预测的地标点的张量
        :param train_batchsize: 类型 int, 训练时的批量大小
        '''
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        # 计算几何信息：Σ(1-cosθn^k)  k=1,2,3
        # 最终得到每张图片的角度权重，形状为 (batch_size, 1)
        attributes_w_n = attribute_gt[:, 1:6].float()
        # 计算属性权重矩阵，其中第1行到第5行分别表示侧脸、正脸、抬头、低头、表情/遮挡等属性。
        # 其元素的数值为{0, 1}表示二分类，即“有无此属性”
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        # 计算每个属性在当前批次中的平均值，亦或者者说频率
        # 因为每个元素的值只能是0（没有）或1（有），因此平均值就代表了当前属性在样本批次中出现的频率
        mat_ratio = torch.Tensor([
            1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio
        ]).to(device)
        # 计算倒数权重，因为频率越低，要求惩罚的权重越大
        # 如果这样不处理，网络会倾向于只学习好占多数的简单样本，而忽略少数困难样本，导致 loss 被简单样本主导。
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)
        # .mul 不是矩阵乘法，而是带有广播机制的按元素相乘
        # 例如，假设 attributes_w_n 的形状是 (batch_size, 5)，mat_ratio 的形状是 (1, 5)，则 mat_ratio 会被广播成 (batch_size, 5)，然后逐元素相乘。
        # 结果就是，在属性矩阵中，如果一个样本具有某个属性，则该属性的标称值从原来的1变为该属性的倒数权重，从而增加了该样本在总损失中的贡献。
        # 每个属性明码标价
        # 然后，将这些加权后的属性值相加，得到每个样本的总属性权重。最终形状： (batch_size, 1)
        # 这一行代码的总体作用就是，为当前 Batch 中的每一张图片，根据它包含的属性，累加计算出该图片的最终 Loss 权重，包含的困难属性越多，越稀有，这张图在计算Loss时所占的比重越大。 

        l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
        # 计算每个样本的地标点 L2 距离的平方和，形状为 (batch_size, 1)，每一行元素的形式为：
        # x1²+y1²+x2²+y2²+...+xN²+yN²
        return torch.mean(weight_angle * weight_attribute * l2_distant), torch.mean(l2_distant)
        # 这里同样不是矩阵乘法，而是按元素相乘，最终第一项得到完整的 Loss 函数值
        # 最后求均值而不是求和，如果是求和的话，Loss 会随着 Batch Size 的增大而增大，导致超参数不稳定
        # 为什么还要返回未加权的 L2 距离平方和的均值呢？这是给人看的，用于监控指标（Metric / Monitoring）。它反映了模型当前预测的坐标和真实坐标平均相差多少。因为第一个 Loss 被权重“污染”了，你无法通过它判断模型到底收敛没有。
        # 也就是：如果只看第一个 Loss，你不知道 Loss 变大是因为模型变差


def smoothL1(y_true, y_pred, beta=1):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    mae = torch.abs(y_true - y_pred)
    loss = torch.sum(torch.where(mae > beta, mae - 0.5 * beta, 0.5 * mae**2 / beta), axis=-1)
    # 大误差用L1 Loss：mae - 0.5 * beta，这部分的梯度是常数（1 或 -1），防止梯度爆炸。当碰到离群点（Outliers）时，不会因为误差非常大而产生巨大的梯度把模型参数打乱
    # 小误差类似L2 Loss：0.5 * mae**2 / beta，这部分的梯度在原点附近是动态减小的（越来越接近 0），能够平滑趋近于零。L1 Loss 在 0 点不可导且梯度始终为 1，容易在最优解附近震荡无法收敛，Smooth L1 解决了这个问题。
    return torch.mean(loss)


def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0, N_LANDMARK=106):
    y_pred = y_pred.reshape(-1, N_LANDMARK, 2)
    y_true = y_true.reshape(-1, N_LANDMARK, 2)
    # 将输入数据恢复成标准的集合形状
    # 神经网络的输出通常是摊平的一维向量，形状为 (batch_size, N_LANDMARK * 2)
    # 本操作将其重新调整为 (batch_size, N_LANDMARK, 2)

    x = y_true - y_pred
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    # 连续性常数：C = w[1-ln(1+w/ε)]
    losses = torch.where(w > absolute_x,
                         w * torch.log(1.0 + absolute_x / epsilon),
                         absolute_x - c)
    # 小误差区间(w > |x|)：wln(1+|x|/ε)，类似于 L2 Loss，在原点附近平滑收敛
    # 大误差区间(|x| >= w)：|x| - C，类似于 L1 Loss，防止梯度爆炸（离群点）
    loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0)
    # torch.sum(losses, axis=[1, 2]): 沿着第1维和第2维求和，即得到每一张图片的总误差
    # torch.mean(..., axis=0): 最后对所有图片的总误差求均值，得到最终的 Loss 值
    return loss