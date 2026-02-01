# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse
# from torch.autograd import Variable
import torch

from models.PFLD import PFLD
from models.PFLD_GhostNet import PFLD_GhostNet
from models.PFLD_GhostNet_Slim import PFLD_GhostNet_Slim
from models.PFLD_GhostOne import PFLD_GhostOne
import copy

import onnx
from onnxsim import simplify
# import onnxoptimizer


def reparameterize_model(model: torch.nn.Module) -> torch.nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.
    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model


parser = argparse.ArgumentParser(description='pytorch2onnx')
parser.add_argument('--model_type', default='PFLD_GhostOne', type=str)
parser.add_argument('--input_size', default=112, type=int)
parser.add_argument('--width_factor', default=1, type=float)
parser.add_argument('--landmark_number', default=98, type=int)
parser.add_argument('--model_path', default="./pfld_ghostone_best.pth")
args = parser.parse_args()

print("=====> load pytorch checkpoint...")
checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
MODEL_DICT = {'PFLD': PFLD,
              'PFLD_GhostNet': PFLD_GhostNet,
              'PFLD_GhostNet_Slim': PFLD_GhostNet_Slim,
              'PFLD_GhostOne': PFLD_GhostOne,
              }
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
MODEL_TYPE = args.model_type
WIDTH_FACTOR = args.width_factor
INPUT_SIZE = args.input_size
LANDMARK_NUMBER = args.landmark_number
model = MODEL_DICT[MODEL_TYPE](WIDTH_FACTOR, INPUT_SIZE, LANDMARK_NUMBER)
model.load_state_dict(new_state_dict)

if 'ghostone' in MODEL_TYPE.lower():
    model = reparameterize_model(model)

print("=====> convert pytorch model to onnx...")
# dummy_input = Variable(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))
dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
# 产生一个随机输入，用随机数填充的 Tensor（张量），它的形状与网络真实的输入完全一致。
# 用途是探路，PyTorch 的计算图是动态的（Dynamic Graph），不运行一次代码就不知道网络结构长什么样。
input_names = ["input"]
# 给 ONNX 模型的输入端口起个名字
# 指定为 ["input"] 后，将来你在其他框架（比如 TensorRT, OpenCV, ncnn）里加载这个模型时，就可以明确地说：“把图片数据喂给名为 'input' 的这个端口”。这对于多输入的模型尤为重要。
output_names = ["output"]
# 给 ONNX 模型的输出端口起个名字。
# 作用同上。方便后续解析结果时，直接通过名称 'output' 拿到预测出的 196 个关键点坐标。
onnx_save_name = "{}_{}_{}.onnx".format(MODEL_TYPE, INPUT_SIZE, WIDTH_FACTOR)
torch.onnx.export(model, dummy_input, onnx_save_name, verbose=False, input_names=input_names, output_names=output_names)
# verbose 如果设为 True，控制台会打印出巨长无比的计算图结构日志（通常只在 DEBUG 时开启）。

model = onnx.load(onnx_save_name)
model_simp, check = simplify(model)
# 它会执行一系列图优化操作，例如：
# 常量折叠 (Constant Folding): 如果网络里有几个算子用来计算一个固定的常数（比如 3 + 5），它会直接算好变成 8，删掉那两个加法节点。
# 算子融合 (Operator Fusion): 把 Conv + BN + Relu 这种经典组合融合成一个超级算子。
# 冗余消除: 去掉那些 Reshape -> Reshape 这种互相抵消的废操作。

assert check, "Simplified ONNX model could not be validated"
# passes = onnxoptimizer.get_fuse_and_elimination_passes()
# opt_model = onnxoptimizer.optimize(model=model, passes=passes)
# final_save_name = "{}_opt.onnx".format(onnx_save_name.split('.')[0])
# onnx.save(opt_model, final_save_name)
# print("=====> ONNX Model save in {}".format(final_save_name))

# Use simplified model as final result since onnxoptimizer is deprecated/incompatible
final_save_name = "{}_sim.onnx".format(onnx_save_name.split('.')[0])
onnx.save(model_simp, final_save_name)
print("=====> ONNX Model save in {}".format(final_save_name))
