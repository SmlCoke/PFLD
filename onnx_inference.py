# -*-coding: utf-8 -*-
import os
import onnxruntime
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import cv2

import argparse
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")


# 裁剪、缩放并进行填充（Letterbox）处理，用于从原图中提取人脸区域并调整为PFLD模型所需的输入尺寸
def cut_resize_letterbox(image, det, target_size):
    # 获取原始图像的宽和高
    iw, ih = image.size

    # 解析人脸检测框的坐标 (x1, y1, x2, y2)
    facebox_x = det[0]
    facebox_y = det[1]
    facebox_w = det[2] - det[0]
    facebox_h = det[3] - det[1]

    # 为了保持长宽比，取宽和高中的较大值作为基准
    facebox_max_length = max(facebox_w, facebox_h)
    
    # 计算填充边缘的长度，使得裁剪区域为正方形
    width_margin_length = (facebox_max_length - facebox_w) / 2
    height_margin_length = (facebox_max_length - facebox_h) / 2

    # 计算扩充后的裁剪区域坐标（正方形）
    face_letterbox_x = facebox_x - width_margin_length
    face_letterbox_y = facebox_y - height_margin_length
    face_letterbox_w = facebox_max_length
    face_letterbox_h = facebox_max_length

    # 计算裁剪区域超出图像边界的大小（处理越界情况）
    top = -face_letterbox_y if face_letterbox_y < 0 else 0
    left = -face_letterbox_x if face_letterbox_x < 0 else 0
    bottom = face_letterbox_y + face_letterbox_h - ih if face_letterbox_y + face_letterbox_h - ih > 0 else 0
    right = face_letterbox_x + face_letterbox_w - iw if face_letterbox_x + face_letterbox_w - iw > 0 else 0

    # 创建一个新的画布，大小包含越界部分，用黑色填充
    margin_image = Image.new('RGB', (iw + right - left, ih + bottom - top), (0, 0, 0))
    # 将原图粘贴到新画布的正确位置
    margin_image.paste(image, (left, top))

    # 从新画布中裁剪出正方形的人脸区域
    face_letterbox = margin_image.crop((face_letterbox_x, face_letterbox_y, face_letterbox_x + face_letterbox_w, face_letterbox_y + face_letterbox_h))
    # 将裁剪出的人脸图像缩放到目标尺寸（如 112x112）
    face_letterbox = face_letterbox.resize(target_size, Image.Resampling.BICUBIC)

    # 返回处理后的人脸图像、缩放比例、以及裁剪区域在原图中的左上角坐标（用于后续坐标还原）
    return face_letterbox, facebox_max_length / target_size[0], face_letterbox_x, face_letterbox_y


# 将 Tensor 转换为 Numpy 数组的辅助函数
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# 在 CPU 上执行非极大值抑制（NMS），用于去除重叠的人脸检测框
def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    # areas: 各个 bounding box 的面积
    scores = dets[:, 4]
    # 这里返回的是索引
    keep = []
    index = scores.argsort()[::-1]
    # 按每个框的置信度 (score) 从高到低对所有候选框进行排序。
    # 原因：我们总是倾向于相信分数最高的那个框是“真命天子”。排好队后，我们永远先挑第一名出来。
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)
        # 直接把当前队伍里的“第一名”（分数最高的框 i）选入结果列表（keep）。
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        # 至此，x11, y11, x22, y22 锁定了 overlap 区域

        w = np.maximum(0, x22 - x11 + 1)  # the widths of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the heights of overlap

        overlaps = w * h
        # overlap 区域面积
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # 交并比

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1
        # 这里加1是因为上面计算ious是基于 areas[index[1:]] 广播的

    return keep


# 处理检测模型的输出，主要涉及坐标转换和置信度筛选
def process_output(dets, thresh, scale, pad_w, pad_h, iw, ih):
    process_dets = []
    # 遍历所有检测结果
    for det in dets:
        # 如果检测置信度小于阈值，则忽略
        if det[4] < thresh:
            continue

        # 解析检测框的中点坐标及宽高
        cx, cy, w, h = det[:4]
        # 将检测框坐标从填充后的图像坐标系转换回原图坐标系
        # (坐标 - padding: 填充偏移量) / 缩放比例
        x1 = max(((cx - w / 2.) - pad_w) / scale, 0.)
        y1 = max(((cy - h / 2.) - pad_h) / scale, 0.)
        x2 = min(((cx + w / 2.) - pad_w) / scale, iw)
        y2 = min(((cy + h / 2.) - pad_h) / scale, ih)
        # 计算最终得分（检测置信度 * 类别概率）
        score = det[4] * det[15]

        process_dets.append([x1, y1, x2, y2, score])

    return process_dets


# 对图像进行 Padding 操作，保持长宽比缩放至目标尺寸
def pad_image(image, target_size):
    iw, ih = image.size
    w, h = target_size

    # 计算缩放比例，取宽和高缩放比中较小的那个，以保证图像完整放入
    scale = min(w / iw, h / ih)
    nw = int(iw * scale + 0.5)
    nh = int(ih * scale + 0.5)

    # 计算需要的 Padding 大小
    pad_w = (w - nw) // 2
    pad_h = (h - nh) // 2
    
    # 缩放图像
    image = image.resize((nw, nh), Image.Resampling.BICUBIC)
    # 创建目标尺寸的新图像，背景填充灰色
    new_image = Image.new('RGB', target_size, (128, 128, 128))

    # 将缩放后的图像粘贴到新图像中心
    new_image.paste(image, (pad_w, pad_h))

    return new_image, scale, pad_w, pad_h


# 将 PIL 图像转换为模型输入的 Tensor
def get_img_tensor(pil_img, use_cuda, target_size, transform):
    iw, ih = pil_img.size
    # 如果图像尺寸不符合目标尺寸，进行缩放
    if iw != target_size[0] or ih != target_size[1]:
        pil_img = pil_img.resize(target_size, Image.Resampling.BICUBIC)

    # 应用预定义的 transforms（如 ToTensor, Normalize）
    tensor_img = transform(pil_img)
    # 增加 batch 维度: (C, H, W) -> (1, C, H, W)
    tensor_img = torch.unsqueeze(tensor_img, 0)
    # 如果使用 CUDA，将 Tensor 移动到 GPU
    if use_cuda:
        tensor_img = tensor_img.cuda()

    return tensor_img


# 对文件夹中的图片进行批量推断
def inference_folder(args):
    # 如果结果保存文件夹不存在，则创建
    if not os.path.exists(args.save_result_folder):
        os.makedirs(args.save_result_folder)

    # 加载 ONNX 模型
    # 加载人脸检测模型
    facedetect_session = onnxruntime.InferenceSession(args.facedetect_onnx_model)
    # 加载 PFLD 关键点检测模型
    pfld_session = onnxruntime.InferenceSession(args.pfld_onnx_model)

    # 定义人脸检测的数据预处理
    detect_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 定义 PFLD 的数据预处理 (归一化到 [-1, 1])
    pfld_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 开始推理
    for img_name in os.listdir(args.test_folder):
        img_path = os.path.join(args.test_folder, img_name)
        # 仅处理 .jpg 和 .png 图片
        if '.jpg' not in img_path and '.png' not in img_path:
            continue

        pil_img = Image.open(img_path)
        iw, ih = pil_img.size

        # ------------------- 人脸检测 -------------------
        # 对图片进行 Padding 和 Letterbox 处理
        # 将图片缩放到(facedetect_input_size, facedetect_input_size)尺寸。如果不满足长宽比，会用灰色像素填充
        pil_img_pad, scale, pad_w, pad_h = pad_image(pil_img, args.facedetect_input_size)
        # 获取 Tensor 输入：缩放、预处理，然后ToTensor，然后搬到 GPU（如果需要）
        detect_tensor_img = get_img_tensor(pil_img_pad, args.use_cuda, args.facedetect_input_size, detect_transform)

        # 运行人脸检测模型
        inputs = {facedetect_session.get_inputs()[0].name: to_numpy(detect_tensor_img)}
        # inputs: dict，内容结构: { "输入节点名称": 输入数据的 Numpy 数组 }
        # facedetect_session.get_inputs()[0].name: 这是从 ONNX 模型文件中读取到的输入层名字（例如 "images" 或 "input"）。ORT 必须通过名字来匹配数据。
        # to_numpy(detect_tensor_img): 这是将 PyTorch Tensor 转换为 Numpy 数组的函数，因为 ONNX Runtime 只能接受 Numpy 数组作为输入。
        # 作用: 告诉推理引擎：“把这块图像数据塞进这个名字的入口”。
        outputs = facedetect_session.run(None, inputs)
        # outputs: list, 包含了一个或多个 Numpy 数组。
        # 参数 None: 第一个参数代表 output_names。传入 None 表示“给我所有的输出节点”。
        # 列表长度: 取决于模型有几个输出头。对于 YOLOv5-Face 这种检测模型，通常导出的 ONNX 只有一个合并后的输出张量，所以列表长度通常为 1。
        preds = outputs[0][0]  # 获取第一个输出
        # outputs[0] 的形状: (1, 25200, 16) (假设是标准的 YOLOv5-Face 640模型)。
        # 1: Batch Size
        # 25200: 每张图像预测的候选框数量
        # 16: 每个候选框的属性（4个坐标(cx, cy, w, h) + 1个置信度（这里有一张人脸的概率是多少?） + 15个类别概率（最后一个概率表示是人脸的概率））
        # output[0][0]: 获取该Batch中的第一张图像的所有候选框数据，形状为 (25200, 16)。

        # 处理输出结果：坐标还原、阈值过滤
        preds = np.array(process_output(preds, 0.5, scale, pad_w, pad_h, iw, ih))
        # 执行 NMS 去除重复框
        keep = py_cpu_nms(preds, 0.5)
        dets = preds[keep]

        # 初始化绘图对象
        draw = ImageDraw.Draw(pil_img)
        # 可选：绘制人脸检测框
        # for det in dets:
        #     draw.rectangle(((det[0], det[1]), (det[2], det[3])), fill=None, outline=(0, 255, 127), width=2)
        # pil_img.save(os.path.join(args.save_result_folder, img_name))

        # ------------------- 关键点检测 -------------------
        for det in dets:
             # 根据检测框裁剪人脸，并调整为 PFLD 输入尺寸
            cut_face_img, scale_l, x_offset, y_offset = cut_resize_letterbox(pil_img, det, args.pfld_input_size)
            # x_offset, y_offset 用于后续坐标还原

            # 获取 PFLD 输入 Tensor
            pfld_tensor_img = get_img_tensor(cut_face_img, args.use_cuda, args.pfld_input_size, pfld_transform)

            # 运行 PFLD 模型
            inputs = {pfld_session.get_inputs()[0].name: to_numpy(pfld_tensor_img)}
            outputs = pfld_session.run(None, inputs)
            preds = outputs[0][0]  # 获取 98 个关键点的坐标 (归一化值)

            # 将关键点坐标映射回原图并绘制
            for i in range(98):
                # 坐标还原：预测坐标 * 输入尺寸 * 缩放比例  + 偏移量
                center_x = preds[i * 2] * args.pfld_input_size[0] * scale_l + x_offset
                center_y = preds[i * 2 + 1] * args.pfld_input_size[1] * scale_l + y_offset
                radius = 1
                # 绘制关键点
                draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), (0, 255, 127))

        # 保存结果图片
        pil_img.save(os.path.join(args.save_result_folder, img_name.split('.')[0] + '.png'))


# 对视频文件进行推断
def inference_video(args):
    # 加载模型
    facedetect_session = onnxruntime.InferenceSession(args.facedetect_onnx_model)
    pfld_session = onnxruntime.InferenceSession(args.pfld_onnx_model)

    detect_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    pfld_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 打开视频文件
    cap = cv2.VideoCapture(args.test_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # 初始化视频写入对象，用于保存结果
    out_cap = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    prev_pts = []  # 用于存储上一帧的关键点，做平滑处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换 BGR (OpenCV) -> RGB (PIL)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        iw, ih = pil_img.size

        # ------------------- 人脸检测 -------------------
        pil_img_pad, scale, pad_w, pad_h = pad_image(pil_img, args.facedetect_input_size)
        detect_tensor_img = get_img_tensor(pil_img_pad, args.use_cuda, args.facedetect_input_size, detect_transform)

        inputs = {facedetect_session.get_inputs()[0].name: to_numpy(detect_tensor_img)}
        outputs = facedetect_session.run(None, inputs)
        preds = outputs[0][0]

        preds = np.array(process_output(preds, 0.5, scale, pad_w, pad_h, iw, ih))
        keep = py_cpu_nms(preds, 0.5)
        dets = preds[keep]

        draw = ImageDraw.Draw(pil_img)
        # for det in dets:
        #     draw.rectangle(((det[0], det[1]), (det[2], det[3])), fill=None, outline=(0, 255, 127), width=2)
        # pil_img.save(os.path.join(args.save_result_folder, img_name))
        
        # ------------------- 关键点检测 -------------------
        for det in dets:
            cut_face_img, scale_l, x_offset, y_offset = cut_resize_letterbox(pil_img, det, args.pfld_input_size)
            # x_offset, y_offset 用于后续坐标还原

            pfld_tensor_img = get_img_tensor(cut_face_img, args.use_cuda, args.pfld_input_size, pfld_transform)

            inputs = {pfld_session.get_inputs()[0].name: to_numpy(pfld_tensor_img)}
            outputs = pfld_session.run(None, inputs)
            preds = outputs[0][0]

            # 如果是第一帧检测到的人脸，初始化 pre_pts
            if len(prev_pts) == 0:
                for i in range(98):
                    center_x = preds[i * 2] * args.pfld_input_size[0] * scale_l + x_offset
                    center_y = preds[i * 2 + 1] * args.pfld_input_size[1] * scale_l + y_offset
                    prev_pts.append(center_x)
                    prev_pts.append(center_y)

            # 平滑处理并绘制点
            for i in range(98):
                center_x = preds[i * 2] * args.pfld_input_size[0] * scale_l + x_offset
                center_y = preds[i * 2 + 1] * args.pfld_input_size[1] * scale_l + y_offset

                # 使用加权平均进行抖动消除 (Exponential Moving Average)
                beta = 0.7
                smooth_center_x = center_x * beta + prev_pts[i * 2] * (1 - beta)
                smooth_center_y = center_y * beta + prev_pts[i * 2 + 1] * (1 - beta)

                # 更新上一帧的关键点记录
                prev_pts[i * 2] = smooth_center_x
                prev_pts[i * 2 + 1] = smooth_center_y

                radius = 4
                draw.ellipse((smooth_center_x - radius, smooth_center_y - radius, smooth_center_x + radius, smooth_center_y + radius), (0, 255, 127))

        # 将处理完的 PIL 图片转换回 OpenCV 格式并写入视频
        cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        out_cap.write(cv_img)

    cap.release()
    out_cap.release()


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    # 是否使用 CUDA 加速 (注意：ONNXRuntime 需要安装对应的 GPU 版本才生效)
    parser.add_argument('--use_cuda', default=True, type=bool)
    # PFLD 关键点检测模型路径
    parser.add_argument('--pfld_onnx_model', default="./onnx_models/PFLD_GhostOne_112_1_opt_sim.onnx", type=str)
    # PFLD 模型输入尺寸
    parser.add_argument('--pfld_input_size', default=(112, 112), type=list)
    # 人脸检测模型路径 (YOLOv5-face)
    parser.add_argument('--facedetect_onnx_model', default="./onnx_models/yolov5face_n_640.onnx", type=str)
    # 人脸检测模型输入尺寸
    parser.add_argument('--facedetect_input_size', default=(640, 640), type=list)
    # 测试图片文件夹路径
    parser.add_argument('--test_folder', default='./test_imgs', type=str)
    # 结果保存路径
    parser.add_argument('--save_result_folder', default='./test_imgs_result', type=str)
    # 测试视频路径
    parser.add_argument('--test_video', default='./test_video/nice.mp4', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # 运行图片文件夹推理
    inference_folder(args)
    # 若要运行视频推理，请取消下方注释
    # inference_video(args)
