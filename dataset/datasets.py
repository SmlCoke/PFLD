import numpy as np
import cv2
import sys
import torch

sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader

# =============================================================================
# 数据增强（Data Augmentation）工具函数
# 以下函数均用于在训练过程中对图像和关键点进行随机变换，以增加模型的鲁棒性
# =============================================================================

def flip(img, annotation):
    """
    对图像和关键点进行水平翻转（左右镜像）。
    
    :param img: 输入图像 (H, W, C)
    :param annotation: 标注信息列表，格式: [x_min, y_min, x_max, y_max, x1, y1, x2, y2, ...]
                       包括前4个 bbox 坐标和后续的 landmark 坐标
    :return: 翻转后的图像和标注
    """
    # 1. 图像水平翻转 (Left-Right Flip)
    img = np.fliplr(img).copy()
    h, w = img.shape[:2]

    # 2. 解析标注数据
    x_min, y_min, x_max, y_max = annotation[0:4]
    landmark_x = annotation[4::2]      # 提取所有偶数索引位置的 x 坐标
    landmark_y = annotation[4 + 1::2]  # 提取所有奇数索引位置的 y 坐标

    # 3. 翻转 BBox 坐标
    # 注意：原本的右边缘翻转后变成左边缘，原本的左边缘变成右边缘
    # 新的 x_min = Width - 旧的 x_max
    bbox = np.array([w - x_max, y_min, w - x_min, y_max])
    
    # 4. 翻转关键点 x 坐标
    for i in range(len(landmark_x)):
        landmark_x[i] = w - landmark_x[i]

    # 5. 重组标注列表
    new_annotation = list()
    new_annotation.append(x_min) # 这里似乎是个BUG？翻转后应该使用 bbox 中的新坐标，但这里append的是旧坐标？
    new_annotation.append(y_min) # 建议核查是否应该 append bbox[0...3]
    new_annotation.append(x_max)
    new_annotation.append(y_max)

    for i in range(len(landmark_x)):
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return img, new_annotation


def channel_shuffle(img, annotation):
    """
    随机通道重排（Channel Shuffle）。
    模拟不同的色彩空间或传感器顺序（如 RGB vs BGR），增强模型对颜色顺序的不敏感性。
    """
    if (img.shape[2] == 3):
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr) # 随机打乱通道顺序
        img = img[..., ch_arr]    # 应用重排
    return img, annotation


def random_noise(img, annotation, limit=[0, 0.2], p=0.5):
    """
    添加随机噪声（椒盐噪声/高斯噪声的简化版）。
    
    :param limit: 噪声强度的范围，默认 [0, 0.2]（相对于 0-255 像素值，这似乎有点小？）
                  看代码逻辑，这里生成的噪声是 uniform(0, 0.2) * 255 -> 0 到 51 之间的随机值
    :param p: 执行概率
    """
    if random.random() < p:
        H, W = img.shape[:2]
        # 生成与图像同尺寸的随机噪声图
        noise = np.random.uniform(limit[0], limit[1], size=(H, W)) * 255

        # 将单通道噪声图扩展为三通道，并叠加到原图上
        # np.newaxis 增加维度 (H, W) -> (H, W, 1)
        # * [1, 1, 1] 广播到 (H, W, 3)
        img = img + noise[:, :, np.newaxis] * np.array([1, 1, 1])
        # 截断值域并在转换回 uint8，防止溢出
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img, annotation


def random_brightness(img, annotation, brightness=0.3):
    """
    随机亮度调整。
    
    :param brightness: 亮度调整因子的这震荡范围，例如 0.3 表示亮度在 [0.7, 1.3] 倍之间变化
    """
    alpha = 1 + np.random.uniform(-brightness, brightness)
    img = alpha * img # 直接对像素值乘系数
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_contrast(img, annotation, contrast=0.3):
    """
    随机对比度调整。
    原理：image = alpha * image + (1 - alpha) * gray_image
         当 alpha > 1，对比度增强；alpha < 1，对比度降低（趋向于灰度平均值）
    """
    coef = np.array([[[0.114, 0.587, 0.299]]])  # RGB 转灰度的加权系数 (Empirical Formula)
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    
    # 计算灰度图
    gray = img * coef
    # 计算整张图的加权平均灰度值（变成一个标量）
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    
    # 调整对比度公式
    img = alpha * img + gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_saturation(img, annotation, saturation=0.5):
    """
    随机饱和度调整。
    原理：在原图和灰度图之间插值。
    """
    coef = np.array([[[0.299, 0.587, 0.114]]]) # 注意：这里的系数与上面对比度函数略有不同，可能是标准不一致
    alpha = np.random.uniform(-saturation, saturation)
    gray = img * coef
    gray = np.sum(gray, axis=2, keepdims=True) # 保持维度 (H, W, 1)
    
    # 在原彩色图和灰度图之间线性插值
    img = alpha * img + (1.0 - alpha) * gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_hue(image, annotation, hue=0.5):
    """
    随机色相（Hue）调整。
    通过转换到 HSV 空间，只修改 H 通道来实现。
    """
    h = int(np.random.uniform(-hue, hue) * 180) # OpenCV 中 H 通道范围是 0-179

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 循环移位 H 通道的值 (Hue 是一个色环)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image, annotation


def scale(img, annotation):
    """
    随机缩放图像（同时缩放标注）。
    """
    f_xy = np.random.uniform(-0.4, 0.8) # 这里生成的因子如果为负数，下面的 resize 可能会报错或产生意外结果？
    # 修正推测：通常缩放因子应为正数，这里可能是作者意图写 1 + uniform(-0.4, 0.8)？
    # 或者这个代码片段是从某个库里摘出来的，需要上文 context 确认 f_xy 的使用方式
    # 假设这里 f_xy 是 scaling factor
    
    origin_h, origin_w = img.shape[:2]

    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    h, w = int(origin_h * f_xy), int(origin_w * f_xy)
    # skimage.transform.resize (假设这里使用了那个库，如果不存则此处为伪代码)
    # 实际上 OpenCV 的 resize 更常用且高效：cv2.resize(img, (w, h))
    image = resize(img, (h, w), preserve_range=True, anti_aliasing=True, mode='constant').astype(np.uint8)

    new_annotation = list()
    for i in range(len(bbox)):
        bbox[i] = bbox[i] * f_xy
        new_annotation.append(bbox[i])

    for i in range(len(landmark_x)):
        landmark_x[i] = landmark_x[i] * f_xy
        landmark_y[i] = landmark_y[i] * f_xy
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return image, new_annotation


def rotate(img, annotation, alpha=30):
    """
    对图像和关键点进行旋转增强。
    
    :param alpha: 旋转角度上限（例如 30 度）
    """
    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]
    
    # 1. 计算旋转中心（BBox 的中心点）
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    # 2. 获取 OpenCV 旋转矩阵
    # 参数：中心点，旋转角度 (angle)，缩放比例 (scale)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    
    # 3. 对图像进行仿射变换（旋转）
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    # 计算 BBox 旋转后的四个顶点坐标
    point_x = [bbox[0], bbox[2], bbox[0], bbox[2]]
    point_y = [bbox[1], bbox[3], bbox[3], bbox[1]]

    new_point_x = list()
    new_point_y = list()
    
    # Bug Warning: 下面的循环用的是 landmark_x, landmark_y 而不是 point_x, point_y？
    # 这段代码看起来像是想计算旋转后的關鍵点，而不是计算旋转后的 BBox 顶点。
    # 如果是计算 landmark，那么 new_annotation.append 的 min/max 逻辑就有点奇怪了，那是用来算新 BBox 的。
    for (x, y) in zip(landmark_x, landmark_y):
        new_point_x.append(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2])
        new_point_y.append(rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2])

    new_annotation = list()
    # 用变换后的所有关键点的最小/最大值作为新的 BBox
    new_annotation.append(min(new_point_x))
    new_annotation.append(min(new_point_y))
    new_annotation.append(max(new_point_x))
    new_annotation.append(max(new_point_y))

    # 保存旋转后的关键点坐标
    for (x, y) in zip(landmark_x, landmark_y):
        new_annotation.append(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2])
        new_annotation.append(rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2])

    return img_rotated_by_alpha, new_annotation


def generate_FT(image):
    """
    生成图像的频域特征图（Fourier Transform Feature Map）。
    可能用于辅助任务，帮助模型学习纹理或高频/低频信息。
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2D 傅里叶变换
    f = np.fft.fft2(image)
    # 频谱中心化（低频移动到图像中心）
    fshift = np.fft.fftshift(f)
    # 转换为对数尺度显示的幅度谱（Log Magnitude Spectrum）
    fimg = np.log(np.abs(fshift) + 1)
    
    # 下面的循环是在做 Min-Max 归一化，但效率极低，建议直接使用 numpy 向量化操作
    # maxx = np.max(fimg); minn = np.min(fimg); fimg = (fimg - minn) / (maxx - minn)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn + 1) / (maxx - minn + 1)
    return fimg


def draw_labelmap(img, pt, sigma=1, type='Gaussian'):
    """
    在热力图（Heatmap）上指定点的位置生成高斯分布。
    用于关键点检测的 Heatmap 回归任务。
    
    :param img: 目标 Heatmap 图像 (单通道)
    :param pt: 关键点坐标 [x, y]
    :param sigma: 高斯分布的标准差，控制热点扩散范围
    """
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # 1. 计算高斯核在图像上的覆盖范围（左上角 ul，右下角 br）
    # 覆盖范围通常取 3*sigma，基本包含了高斯分布 99% 的能量
    ul = [int(int(pt[0]) - 3 * sigma), int(int(pt[1]) - 3 * sigma)]
    br = [int(int(pt[0]) + 3 * sigma + 1), int(int(pt[1]) + 3 * sigma + 1)]
    
    # 边界检查：如果整个高斯核都跑出图像外了，直接返回
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img) # 注意：这一行调用了一个未定义的函数 to_torch

    # 2. 生成高斯核
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # 3. 处理边界裁剪（如果是部分跑出图像外）
    # Usable gaussian range: 高斯核中有效的部分
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range: 图像中被覆盖的区域
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    # 4. 将高斯核赋值给图像对应的一块 patch
    # 注意：这里直接赋值会覆盖原来的值，如果两个关键点很近，后画的会覆盖先画的。
    # 这种做法在密集点时可能有问题，通常应该取 max()
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


class WLFWDatasets(data.Dataset):
    """
    PyTorch Dataset 类，用于加载 WFLW 格式的数据集。
    """
    def __init__(self, file_list, transforms=None):
        self.line = None
        self.lm_number = 98 # 关键点数量
        # self.img_size = 96  # 注意：这里写死了 96，但预处理脚本里是 112，需确认一致性
        # self.ft_size = self.img_size // 2
        # self.hm_size = self.img_size // 2
        # fj 于202601291255注释
        self.transforms = transforms
        
        # 读取标注列表文件 (list.txt)
        # 每行格式: /path/to/img.png x1 y1 x2 y2 ...
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        """
        魔术方法，规定如何读取第 i 个样本
        """
        self.line = self.lines[index].strip()
        
        # 1. 解析图片路径
        # 假设文件名以 .png 结尾，以此分割路径和后续坐标
        # 假设这一行是： /root/autodl-tmp/PFLD_data/test_data_repeat80/imgs/0_37_Soccer_soccer_ball_37_45_0.png 0.08623407242145945 0.20157444730718085 0.09697358151699634 0.2529571208548039 0.10821752345308344 0.3042313596035572 0.1204690324499252 0.3552731453104222 0.13421914932575632 0.40593070172249 0.14998017980697306  ...  
        jpg_idx = self.line.find('png')
        line_data = [self.line[:jpg_idx + 3]] # 第一个元素：图片路径
        line_data.extend(self.line[jpg_idx + 4:].split()) # 后续元素：坐标字符串列表
        self.line = line_data

        self.img = cv2.imread(self.line[0])
        # 打开图片

        # generate ft (傅里叶特征，被注释掉了)
        # self.ft_img = generate_FT(self.img)
        # self.ft_img = cv2.resize(self.ft_img, (self.ft_size, self.ft_size))
        # self.ft_img = torch.from_numpy(self.ft_img).float()
        # self.ft_img = torch.unsqueeze(self.ft_img, 0)

        # 2. 解析关键点坐标
        # 从 line_data[1] 开始到 [197] (对应 98*2 个坐标)
        self.landmark = np.asarray(self.line[1:197], dtype=np.float32)

        # generate heatmap (热力图生成，被注释掉了)
        # 这里的逻辑是为每个关键点生成一张单独的热力图，或者合并成一张
        # self.heatmaps = np.zeros((self.lm_number, self.img_size, self.img_size))
        # for idx in range(self.lm_number):
        #     self.heatmaps[idx, :, :] = draw_labelmap(self.heatmaps[idx, :, :], (self.landmark[idx * 2] * self.img_size, self.landmark[idx * 2 + 1] * self.img_size))
        # ...

        # 3. 数据增强变换 (transforms)
        # 注意：这里的 transforms 应该同时处理 img 和 landmark，
        # 如果 self.transforms 只是 torchvision 的 transform，会导致 landmark 和 img 不匹配！
        if self.transforms:
            self.img = self.transforms(self.img)
            # 根据 config.py 记载，这里的 transforms 只包含 ToTensor 和 Normalize，
            
        return self.img, self.landmark

    def __len__(self):
        '''
        __len__ 魔术方法，返回数据集大小，即 Total Samples 数量
        '''

        return len(self.lines)


if __name__ == '__main__':
    file_list = './data/test_data/list.txt'
    wlfwdataset = WLFWDatasets(file_list)
    dataloader = DataLoader(wlfwdataset, batch_size=256, shuffle=True, num_workers=0, drop_last=False)
    for img, landmark, attribute, euler_angle in dataloader:
        print("img shape", img.shape)
        print("landmark size", landmark.size())
        print("attrbute size", attribute)
        print("euler_angle", euler_angle.size())
