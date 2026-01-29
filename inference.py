import argparse
import os
import cv2
import numpy as np
import torch
import time  # <--- 将 import time 移动到这里，作为全局导入
from torchvision import transforms
import torch.backends.cudnn as cudnn

from models.PFLD import PFLD
from models.PFLD_GhostNet import PFLD_GhostNet
from models.PFLD_GhostNet_Slim import PFLD_GhostNet_Slim
from models.PFLD_GhostOne import PFLD_GhostOne

# 启用 Cudnn 加速
cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True

def inference_single_image(model, img_path, args, transform):
    """
    对单张图片进行推理
    """
    # 1. 原始读取 (保留原始尺寸用于最终画图)
    origin_img = cv2.imread(img_path)
    if origin_img is None:
        print(f"Error: 无法读取图片 {img_path}")
        return
    
    # 获取原始图片的宽高
    origin_h, origin_w, _ = origin_img.shape

    # 2. 预处理 (Resize -> ToTensor -> Normalize)
    # 必须缩放到模型要求的输入尺寸 (例如 112x112)
    img = cv2.resize(origin_img, (args.input_size, args.input_size))
    # 转换并将 BGR [0, 255] -> RGB -> Tensor [0.0, 1.0] -> Normalize [-1.0, 1.0]
    # 注意：cv2 读取的是 BGR，transforms.ToTensor 会保持通道顺序，通常训练时如果用的 cv2 读取，推理也用 cv2 即可
    img_tensor = transform(img)
    # 增加 Batch 维度: [3, 112, 112] -> [1, 3, 112, 112]
    img_tensor = img_tensor.unsqueeze(0).to(args.device)

    # 3. 模型推理
    with torch.no_grad():
        start = time.time()  # <--- 这里现在可以正确访问到 time 模块了
        landmarks = model(img_tensor)
        end = time.time()
        print(f"Inference time: {(end-start)*1000:.2f}ms")

    # 4. 后处理 (坐标还原)
    # 将 Tensor 转回做 Numpy (BatchSize, 98*2)
    landmarks = landmarks.cpu().numpy()
    # 变形为 (98, 2)
    landmarks = landmarks.reshape(-1, 2)

    # 关键点反归一化：
    # 模型输出的是 0~1 之间的相对坐标。
    # 如果想画在原图上，需要乘以原图的宽高 (origin_w, origin_h)
    landmarks[:, 0] = landmarks[:, 0] * origin_w
    landmarks[:, 1] = landmarks[:, 1] * origin_h

    # 5. 可视化绘制
    # 在原图上画点
    for (x, y) in landmarks:
        cv2.circle(origin_img, (int(x), int(y)), 2, (0, 0, 255), -1)

    # 6. 保存结果
    filename = os.path.basename(img_path)
    save_path = os.path.join(args.output_folder, filename)
    cv2.imwrite(save_path, origin_img)
    print(f"Saved result: {save_path}")


def main(args):
    # 1. 检查输出目录
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # 2. 加载模型架构
    MODEL_DICT = {'PFLD': PFLD,
                  'PFLD_GhostNet': PFLD_GhostNet,
                  'PFLD_GhostNet_Slim': PFLD_GhostNet_Slim,
                  'PFLD_GhostOne': PFLD_GhostOne,
                  }
    
    print(f"Loading model: {args.model_type}...")
    model = MODEL_DICT[args.model_type](args.width_factor, args.input_size, args.landmark_number).to(args.device)

    # 3. 加载权重
    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=args.device)
        
        # 获取 state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # === 修复前缀问题 ===
        # 如果训练时使用了 torch.compile，保存的权重可能会带有 "_orig_mod." 前缀
        # 即使只用了 DataParallel/DistributedDataParallel，也可能带有 "module." 前缀
        # 这里统一进行清理，确保加载到纯净模型中
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            # 去除 "_orig_mod." 前缀 (torch.compile)
            if k.startswith('_orig_mod.'):
                k = k[10:]
            # 去除 "module." 前缀 (DataParallel)
            if k.startswith('module.'):
                k = k[7:]
                
            new_state_dict[k] = v

        try:
            model.load_state_dict(new_state_dict)
            print("Model loaded successfully.")
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            print("Trying strict=False load...")
            model.load_state_dict(new_state_dict, strict=False)
            print("Loaded with strict=False (some keys might be missing).")
            
    else:
        print(f"Error: 模型文件不存在 {args.model_path}")
        return

    model.eval()

    # 4. 定义预处理流程 (需与训练保持一致)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 5. 遍历文件夹进行推理
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in os.listdir(args.input_folder) if os.path.splitext(f)[1].lower() in valid_extensions]

    if not image_files:
        print(f"No images found in {args.input_folder}")
        return

    print(f"Found {len(image_files)} images. Starting inference...")
    
    for img_name in image_files:
        img_path = os.path.join(args.input_folder, img_name)
        inference_single_image(model, img_path, args, transform)

    print("All Done.")


def parse_args():
    parser = argparse.ArgumentParser(description='Inference Script')
    
    # 必要的路径参数
    parser.add_argument('--input_folder', default='./test_imgs', type=str, help='输入图片的文件夹路径')
    parser.add_argument('--output_folder', default='./test_results', type=str, help='保存结果的文件夹路径')
    parser.add_argument('--model_path', default="./pfld_ghostnet_slim_best.pth", type=str, help='模型权重路径')
    
    # 模型配置参数
    parser.add_argument('--model_type', default='PFLD_GhostNet_Slim', type=str)
    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--width_factor', default=1, type=float)
    parser.add_argument('--landmark_number', default=98, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='cuda or cpu')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
