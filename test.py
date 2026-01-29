import argparse
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
# from scipy.integrate import simps
from scipy.integrate import simpson as simps

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.datasets import WLFWDatasets

from models.PFLD import PFLD
from models.PFLD_GhostNet import PFLD_GhostNet
from models.PFLD_GhostNet_Slim import PFLD_GhostNet_Slim
from models.PFLD_GhostOne import PFLD_GhostOne

cudnn.benchmark = True
cudnn.determinstic = True

cudnn.enabled = True


def compute_nme(preds, target):
    """ 计算 NME (Normalized Mean Error) 评价指标
    
    Args:
        preds:  预测的关键点坐标, numpy array, shape is (N, L, 2)
        target: 真实的所谓标签坐标, numpy array, shape is (N, L, 2)
                N: batchsize (Batch大小)
                L: num of landmark (关键点数量)
    """
    N = preds.shape[0] # 图片数量
    L = preds.shape[1] # 关键点数量
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i,], target[i,]
        # 根据数据集类型，选择不同的归一化标准（通常是两眼间距或外眼角间距）
        # 为什么要归一化？为了让误差指标与人脸在图片中的绝对大小无关。
        # 否则图片越大，绝对像素误差自然越大，不公平。
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8,] - pts_gt[9,])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36,] - pts_gt[45,])
        elif L == 98: # WFLW 数据集
            # 使用第 60 号点（左眼角）和第 72 号点（右眼角）的距离作为分母
            interocular = np.linalg.norm(pts_gt[60,] - pts_gt[72,])
        else:
            print(L)
            raise ValueError('Number of landmarks is wrong')
        
        # 公式：误差 = sum(每个点的欧式距离) / (关键点数 * 归一化距离)
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def compute_auc(errors, failureThreshold, step=0.0001, showCurve=True):
    """
    计算 AUC (Area Under Curve) 和 Failure Rate (失败率)
    
    Args:
        errors: 包含每个样本 NME 误差的列表
        failureThreshold: 判定为“失败”的阈值（例如 0.1，即误差超过 10% 算检测失败）
    """
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    # CED (Cumulative Error Distribution) 累积误差分布
    # 计算有多少比例的样本，其误差小于 x
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    # 使用 Simpson 积分法计算曲线下的面积
    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1] # 误差超过阈值的样本比例

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate


def validate(model, wlfw_val_dataloader, args):
    """
    模型验证/测试的主逻辑
    """
    # 切换为评估模式（固定BN，禁用Dropout）
    model.eval()

    nme_list = []
    cost_time = []
    
    # 禁用梯度计算，节省显存并加速
    with torch.no_grad():
        idx = 0
        # 遍历测试集
        for img, landmark_gt in wlfw_val_dataloader:
            img = img.to(args.device)
            landmark_gt = landmark_gt.to(args.device)

            # 记录推理时间
            start_time = time.time()
            landmarks = model(img)
            cost_time.append(time.time() - start_time)

            # 将 Tensor 转换回 Numpy
            # reshape 为 (N, 98, 2)
            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()

            # 可视化功能 (如果参数开启)
            if args.show_image:
                # 反归一化：(x * 0.5 + 0.5) 将 [-1, 1] 变换回 [0, 1]
                # transpose: (C, H, W) -> (H, W, C)
                show_img = np.array(np.transpose((img[0] * 0.5 + 0.5).cpu().numpy(), (1, 2, 0)))
                show_img = (show_img * 255).astype(np.uint8)
                np.clip(show_img, 0, 255)

                # 关键点坐标反归一化：从 [0, 1] 映射回图像尺寸 (例如 112)
                pre_landmark = landmarks[0, :, :2] * [args.input_size, args.input_size]

                # 这是一个不太优雅的临时文件写入方式，但为了用 cv2.imread 重新读取纯净的 BGR 数据
                cv2.imwrite("xxx.jpg", show_img)
                img_clone = cv2.imread("xxx.jpg")

                # 画点
                for ptidx, (x, y) in enumerate(pre_landmark):
                    cv2.circle(img_clone, (int(x), int(y)), 1, (0, 0, 255), -1)
                # 保存可视化结果
                cv2.imwrite("xx_{}.jpg".format(idx), img_clone)

            # 计算该 Batch 所有图片的 NME
            nme_temp = compute_nme(landmarks, landmark_gt[:, :, :2])
            for item in nme_temp:
                nme_list.append(item)
            idx += 1
            
        # 打印最终统计指标
        # 1. 平均 NME
        print('nme: {:.4f}'.format(np.mean(nme_list)))
        
        # 2. AUC 和 失败率
        failureThreshold = 0.1
        auc, failure_rate = compute_auc(nme_list, failureThreshold)
        print('auc @ {:.1f} failureThreshold: {:.4f}'.format(failureThreshold, auc))
        print('failure_rate: {:}'.format(failure_rate))
        
        # 3. 平均每张图的推理耗时
        print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))


def main(args):
    # 定义模型字典，根据名称选择对应的类
    MODEL_DICT = {'PFLD': PFLD,
                  'PFLD_GhostNet': PFLD_GhostNet,
                  'PFLD_GhostNet_Slim': PFLD_GhostNet_Slim,
                  'PFLD_GhostOne': PFLD_GhostOne,
                  }
    
    # 1. 实例化模型
    MODEL_TYPE = args.model_type
    WIDTH_FACTOR = args.width_factor
    INPUT_SIZE = args.input_size
    LANDMARK_NUMBER = args.landmark_number
    model = MODEL_DICT[MODEL_TYPE](WIDTH_FACTOR, INPUT_SIZE, LANDMARK_NUMBER).to(args.device)

    # 2. 加载预训练权重
    checkpoint = torch.load(args.model_path, map_location=args.device)
    # 如果 checkpoint 包含 'state_dict' 键（通常是保存了整个 checkpoint 字典），则加载该键
    # 否则直接加载（如果直接保存的是 state_dict）
    # 这里代码直接 load，因为在 utils.py 中，保存模型时直接采用的是torch.save(model.state_dict(), os.path.join(save_path, ('{}_step:{}.pth'.format(cfg.MODEL_TYPE.lower(), step))))。
    # 这意味着当你加载时，torch.load(path) 返回的直接就是那个字典。
    # 建议加上: if 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'])

    model.load_state_dict(checkpoint)

    # 3. 准备测试数据
    # 标准化：公式：x' = (x - mean)/sigma = (x-0.5)/0.5
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    wlfw_val_dataset = WLFWDatasets(args.test_dataset, transform)
    # 验证时 batch_size 设为 1，方便逐张计算和可视化
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset, batch_size=1, shuffle=False, num_workers=8)

    # 4. 执行验证
    validate(model, wlfw_val_dataloader, args)


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_type', default='PFLD_GhostNet_Slim', type=str)
    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--width_factor', default=1, type=float)
    parser.add_argument('--landmark_number', default=98, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--model_path', default="./pfld_ghostnet_slim_best.pth", type=str)
    parser.add_argument('--test_dataset', default='./data/test_data_repeat80/list.txt', type=str)
    # 是否在运行过程中把带有关键点的结果图保存下来（会在当前目录生成 xx_0.jpg, xx_1.jpg...）
    parser.add_argument('--show_image', default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
