def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle,
                landmarks, train_batchsize):
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
        # 其元素的数值为{0, 1}表示二分类，即“有无此属性”
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        # 计算每个属性在当前批次中的平均值，亦或者者说频率
        mat_ratio = torch.Tensor([
            1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio
        ]).to(device)
        # 计算每个属性的权重，权重 = 1/频率成
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)
        # (batch_size, 1)
        l2_distant = torch.sum(
            (landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
        
        return torch.mean(weight_angle * weight_attribute * l2_distant), torch.mean(l2_distant)
        