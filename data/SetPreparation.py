# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import shutil
import sys

# 调试模式开关，如果为 True，只处理前 100 张图片，用于快速测试脚本逻辑
debug = False


def rotate(angle, center, landmark):
    """
    对关键点进行旋转变换。
    
    :param angle: 旋转角度（度数）。
    :param center: 旋转中心坐标 (x, y)。
    :param landmark: 关键点坐标数组，形状为 (N, 2)。
    :return: 
        M: 旋转矩阵 (2x3)。
        landmark_: 旋转后的关键点坐标。
    """
    # 将角度转换为弧度
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    
    # 构建 2x3 的仿射变换旋转矩阵 M
    # 在计算机图像处理中，通常左上角为原点，Y轴向下。此时坐标旋转给公式：
    # x' = xcosθ - ysinθ, y' = xsinθ + ycosθ
    # 中的 θ 为正时，代表顺时针旋转。
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1 - alpha) * center[0] - beta * center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta * center[0] + (1 - alpha) * center[1]

    # 对关键点进行矩阵乘法，计算旋转后的坐标
    landmark_ = np.asarray([(M[0, 0] * x + M[0, 1] * y + M[0, 2],
                             M[1, 0] * x + M[1, 1] * y + M[1, 2]) for (x, y) in landmark])
    return M, landmark_


class ImageDate():
    def __init__(self, line, imgDir, image_size=112):
        self.image_size = image_size
        line = line.strip().split()
        # .strip() 去掉首尾空格，.split() 按空格分割
        # WFLW 数据集标注行内容说明（如下序号代表第几个元素）：
        # 0-195: 98个关键点的坐标 (x, y)，共 98*2 = 196 个数值
        # 196-199: bbox 坐标点 (x1, y1, x2, y2)
        # 200: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        # 201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        # 202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        # 203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        # 204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        # 205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        # 206: 图片名称 (相对于图片目录的相对路径)
        assert (len(line) == 207) 
        # 每一行只可能有207个元素，否则报错
        self.list = line
        self.landmark = np.asarray(list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
        # map(float, line[:196]) 将前196个字符串转换为浮点数
        # dtype=np.float32 指定数据类型为32位浮点数
        # reshape(-1, 2) 将一维数组重塑为二维数组，每行2个元素（x, y）,-1表示自动推断行数，最终数据格式(98, 2)
        self.box = np.asarray(list(map(int, line[196:200])), dtype=np.int32)
        # map(int, line[196:200]) 将 bbox 坐标字符串转换为整数（这个整数可能是坐标/像素索引）
        self.path = os.path.join(imgDir, line[206])
        # imgDir = ./WFLW/WFLW_images
        # line[206] 示例： 0--WFLW_images--AFW--AF
        self.img = None

        self.imgs = []      # 存储处理后的图片列表（包含原图和增强后的图片）
        self.landmarks = [] # 存储对应的归一化后的关键点
        self.boxes = []     # (代码中未使用)

    def load_data(self, is_train, repeat, mirror=None):
        """
        加载并处理数据：裁剪人脸，归一化，以及进行数据增强。
        
        :param is_train: 是否为训练集（训练集会进行数据增强）。
        :param repeat: 训练集样本扩充的目标数量（通过增强生成 repeat 张图片）。
        :param mirror: 镜像文件路径，用于左右翻转时交换对称的关键点索引。
        """
        # 1. 加载镜像索引文件（如果提供了）
        if mirror is not None:
            with open(mirror, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))
        
        # 2. 计算包含所有关键点的最小包围盒
        xy = np.min(self.landmark, axis=0).astype(np.int32)
        # 分别求出 x 和 y 的最小值，作为包围盒的左上角坐标，axis = 0 表示按列计算
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        # 分别求出 x 和 y 的最大值，作为包围盒的右下角坐标， axis = 0 表示按列计算
        wh = zz - xy + 1
        # wh = (width, height)

        # 3. 计算人脸中心和裁剪尺寸
        center = (xy + wh / 2).astype(np.int32)
        img = cv2.imread(self.path)
        # 将裁剪框稍微扩大 1.2 倍，保证完整的人脸被包含
        boxsize = int(np.max(wh) * 1.2) 
        # 取宽、高的最大值并扩大 1.2 倍，作为裁剪框的边长
        xy = center - boxsize // 2
        x1, y1 = xy
        # 更新新的左上角坐标
        x2, y2 = xy + boxsize
        height, width, _ = img.shape
        
        # 4. 处理边界情况（如果裁剪框超出图像边缘，需要 padding）
        dx = max(0, -x1) # 左边超出的量
        dy = max(0, -y1) # 上边超出的量
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width) # 右边超出的量
        edy = max(0, y2 - height) # 下边超出的量
        x2 = min(width, x2)
        y2 = min(height, y2)

        # 裁剪图像
        imgT = img[y1:y2, x1:x2]
        # 如果有超出部分，使用黑色 (0) 填充 (Padding)
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            # imgT 表示输入图像，dy, edy, dx, edx 分别表示上、下、左、右需要填充的像素行/列数。
            # cv2.BORDER_CONSTANT 表示使用常数值填充，0 表示填充值为黑色。
            # 最终效果就是保证裁剪出来的图像大小为 boxsize x boxsize，即使原图边界不够也能补齐。
        
        # 异常检测：如果裁剪出的图像为空，显示关键点并退出（用于调试坏数据）
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # 将图像转化为彩色(BGR)
            for x, y in (self.landmark + 0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
                # 先将 Landmark 坐标用四舍五入取整得到像素坐标，然后画半径为1的红色(0,0,255)实心圆点
            cv2.imshow('0', imgTT)
            # 弹出一个窗口展示这张带有红色关键点的“问题图片”。
            if cv2.waitKey(0) == 27:
                # 程序无限暂停，直到用户按下一个键盘按键。
                # 27 表示键盘上的 ESC 键
                # 如果用户按下 ESC 键，程序将调用 exit() ，整个脚本终止运行
                # 如果用户按下其他键，程序继续执行
                exit()
        
        # 5. 调整图像大小到目标尺寸 (如 112x112)
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        
        # 6. 关键点坐标归一化 (Normalize landmarks to [0, 1])
        # 将绝对坐标转换为相对于裁剪框的相对坐标
        landmark = (self.landmark - xy) / boxsize
        # 此时的 xy 的含义是：原图的 xy 点，就是裁剪框的左上角
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        
        # 添加基础样本（无增强）
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

        # 7. 数据增强 (Data Augmentation) - 仅针对训练集
        if is_train:
            # 持续进行增强，直到达到指定的样本数量 (repeat)
            while len(self.imgs) < repeat:
                # 随机旋转 (-30 到 30 度)
                angle = np.random.randint(-30, 30)
                # 随机扰动中心点 (Shift)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                # 计算旋转矩阵并旋转关键点
                M, landmark = rotate(angle, (cx, cy), self.landmark)
                
                # 对原图进行仿射变换 (Rotation)
                # 注意：这里先把图像放大 1.1 倍读入，防止旋转后出现过多黑边
                # 这里通过 * 1.1 预防性地把画布扩大了 10%。确保旋转后的人脸依然大概率完整保留在画面中心。
                imgT = cv2.warpAffine(img, M, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))

                # 随机缩放 (Scale)
                # 计算旋转后关键点的新范围
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                # ptp: peak to peak，计算每一列(axis = 0)的最大值与最小值之差，从而求出人脸区域的(width, height)
                # 随机选择裁剪框大小，实现缩放效果
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                # 裁剪框边长最小为 wh 的最小值，最大为 wh 的 1.25 倍
                
                # 基于新的中心和大小计算裁剪区域
                xy = np.asarray((cx - size // 2, cy - size // 2), dtype=np.int32)
                # xy 表示裁剪框的左上角坐标
                # 归一化新的关键点
                landmark = (landmark - xy) / size
                
                # 过滤掉关键点跑出图片范围的无效增广样本
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                # 再次执行裁剪和 Padding 逻辑
                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                # 随机水平翻转 (Random Horizontal Flip)
                if mirror is not None and np.random.choice((True, False)):
                    # x 坐标翻转: x_new = 1 - x_old
                    landmark[:, 0] = 1 - landmark[:, 0]
                    # 关键点索引对称交换 (例如左眼变右眼)
                    landmark = landmark[mirror_idx]
                    # 图片翻转
                    imgT = cv2.flip(imgT, 1)
                
                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        """
        保存处理后的图片和标签。
        
        :param path: 保存图片的目录。
        :param prefix: 图片文件名前缀 (e.g., "0_file_name").
        :return: 标签行的列表。
        """
        labels = []
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            # 确保关键点维度正确
            assert lanmark.shape == (98, 2)
            # 构造保存路径
            save_path = os.path.join(path, prefix + '_' + str(i) + '.png')
            assert not os.path.exists(save_path), save_path
            # 如果文件已存在则报错，防止覆盖
            cv2.imwrite(save_path, img)

            # 将关键点展平并转换为字符串: "x1 y1 x2 y2 ..."
            landmark_str = ' '.join(list(map(str, lanmark.reshape(-1).tolist())))

            # 构造新标签格式: 图片路径 + 关键点坐标
            label = '{} {}\n'.format(save_path, landmark_str)

            labels.append(label)
        return labels


def get_dataset_list(imgDir, outDir, landmarkDir, is_train, repeat_times, Mirror_file):
    """
    处理整个数据集的主循环。
    
    :param imgDir: WFLW 原图所在目录。
    :param outDir: 输出目录 (train_data_repeatXX 或 test_data_repeatXX)。
    :param landmarkDir: 原始标注文件路径 (.txt)。
    :param is_train: 是否为训练模式 (决定是否进行增强)。
    :param repeat_times: 增强倍数。
    :param Mirror_file: 镜像索引文件路径。
    """
    with open(landmarkDir, 'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        if debug:
            lines = lines[:100]
        
        # 遍历每张图片进行处理
        for i, line in enumerate(lines):
            Img = ImageDate(line, imgDir)
            img_name = Img.path
            # 加载并处理数据 (裁剪 + 增强)
            Img.load_data(is_train, repeat_times, Mirror_file)
            # 这里的 Mirror_file 就是 if __name__ ==  "__mainn__", 模块中定义的全局Mirror_file，因为 Python 中的 if, for 语句块不会创建新的作用域，在 if __name__ == '__main__': 之中定义的变量依然是全局变量。因此当脚本执行到这一行时，解释器现在局部作用域(get_dataset_list 函数)中找不到 Mirror_file 变量，就会去全局作用域(模块级别)中查找，找到了就使用它。
            # 虽然代码能跑，但这通常被认为是不规范的编程习惯 (Bad Practice)，最标准的做法是将 Mirror_file 显式地作为参数传递给 get_dataset_list 函数。

            _, filename = os.path.split(img_name)
            # 案例：D:\CICIEC\Project\PFLD_GhostOne\data\WFLW\WFLW_images\1--Handshaking\1_Handshaking_Handshaking_1_35.jpg -> filename = 1_Handshaking_Handshaking_1_35.jpg
            filename, _ = os.path.splitext(filename)
            # 去掉扩展名，filename = 1_Handshaking_Handshaking_1_35

            # 保存数据到磁盘，并获取标签字符串
            label_txt = Img.save_data(save_img, str(i) + '_' + filename)
            labels.append(label_txt)
            
            # 打印进度
            if ((i + 1) % 100) == 0:
                print('file: {}/{}'.format(i + 1, len(lines)))

    # 将生成的标签列表写入 list.txt
    with open(os.path.join(outDir, 'list.txt'), 'w') as f:
        for label in labels:
            f.writelines(label)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__)) # 当前脚本所在目录 (data/)
    imageDirs = './WFLW/WFLW_images'                       # 原图目录 (相对路径)
    Mirror_file = './WFLW/WFLW_annotations/Mirror98.txt'   # 镜像索引文件

    # 原始标注列表文件 (测试集和训练集)
    landmarkDirs = ['./WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt',
                    './WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt']

    current_directory = os.path.dirname(os.path.abspath(__file__))

    repeat_times = 80 # 训练集每张图扩充到的数量
    
    # 定义输出目录
    outDirs = ['{}/test_data_repeat{}'.format(current_directory, repeat_times),
               '{}/train_data_repeat{}'.format(current_directory, repeat_times)]
               
    for landmarkDir, outDir in zip(landmarkDirs, outDirs):
        outDir = os.path.join(root_dir, outDir)
        print(outDir)
        # 如果输出目录已存在，先删除，重新生成
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        os.mkdir(outDir)
        
        # 根据文件名判断是训练集还是测试集
        if 'list_98pt_rect_attr_test.txt' in landmarkDir:
            is_train = False
        else:
            is_train = True
            
        # 执行处理
        get_dataset_list(imageDirs, outDir, landmarkDir, is_train, repeat_times=repeat_times, Mirror_file = Mirror_file)
    print('end')
