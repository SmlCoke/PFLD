#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import warnings
import random
import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from dataset.datasets import WLFWDatasets
from utils.utils import init_weights, save_checkpoint, set_logger, write_cfg
from utils.loss import LandmarkLoss
from test import compute_nme

from models.PFLD import PFLD
from models.PFLD_GhostNet import PFLD_GhostNet
from models.PFLD_GhostNet_Slim import PFLD_GhostNet_Slim
from models.PFLD_GhostOne import PFLD_GhostOne

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

warnings.filterwarnings("ignore")


def train(model, train_dataloader, loss_fn, optimizer, cfg, progress, batch_task):
    """
    执行一个 Epoch 的训练过程。
    
    Args:
        model: 模型实例
        train_dataloader: 训练集的数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        cfg: 全局配置对象
        progress: rich 进度条实例
        batch_task: 进度条的任务 ID
    """
    losses = []
    # 【重要】切换模型到训练模式
    # 这会启用 BatchNorm 层的统计更新和 Dropout 层的随机丢弃行为
    model.train()

    for img, landmark_gt in train_dataloader:
        # for in 自动调用train_dataloader的成员train_dataset的魔术方法 __getitem__，疯狂调用 Batch Size 次，然后打包(Collate_fn)，让 DataLoader 拿到两个巨大的张量 Image Tensor (Batch Size, 3, 112, 112) 和 Landmark Tensor (Batch Size, 196)，讲这两个张量返回给 for 循环
        # 此外，由于设置了 pin_memory=True，DataLoader 会在后台把数据预先加载到 GPU 的固定内存区域，提升数据传输速度
        # num_workers 设置为 n，DataLoader 会启动 n 个子进程并行读取数据（__getitem__），进一步加快数据加载速度，为 for 循环拼出一个 Batch 的数据
        # prefetch_factor 设置为 m，DataLoader 会让每个子进程提前准备 m 个 Batch 的数据，确保数据源源不断地供应给 for 循环
        progress.advance(batch_task, advance=1)

        img = img.to(cfg.DEVICE)
        landmark_gt = landmark_gt.to(cfg.DEVICE)
        
        # 1. 前向传播：将图片输入模型，获取预测的关键点
        landmark_pred = model(img)
        
        # 2. 计算损失：比较预测值与真实标签的差异
        loss = loss_fn(landmark_gt, landmark_pred)
        
        # 3. 反向传播三步曲：
        optimizer.zero_grad() # a. 清空上一轮迭代遗留的梯度信息
        loss.backward()       # b. 反向传播，计算当前参数的梯度
        optimizer.step()      # c. 根据梯度更新模型参数

        losses.append(loss.cpu().detach().numpy())
        # .cpu(): 将张量从 GPU 移动到 CPU（如果在 GPU 上的话）
        # .detach(): 从计算图中分离张量，防止梯度继续传播
        # .numpy(): 转换为 Numpy 数组，方便后续处理

    return np.mean(losses)
    # 注意这里返回的是所有 Batch 的平均损失值


def validate(model, val_dataloader, loss_fn, cfg, progress, test_task):
    """
    执行验证过程，评估模型在验证集上的泛化能力。
    """
    # 【重要】切换模型到评估模式
    # 这会锁定 BatchNorm 的均值方差，并禁用 Dropout，保证推理结果的一致性
    model.eval()
    losses = []
    nme_list = []
    progress.reset(test_task)
    progress.update(test_task, total=len(val_dataloader))

    # 【重要】禁用梯度计算
    # 验证阶段不需要反向传播，关闭梯度计算可以大幅减少显存占用并加速推理
    with torch.no_grad():
        for img, landmark_gt in val_dataloader:
            progress.advance(test_task, advance=1)
            img = img.to(cfg.DEVICE)
            landmark_gt = landmark_gt.to(cfg.DEVICE)
            
            # 1. 计算损失（用于监控模型收敛情况）
            landmark_pred = model(img)
            loss = loss_fn(landmark_gt, landmark_pred)
            losses.append(loss.cpu().numpy())

            # 2. 计算 NME (Normalized Mean Error) 评价指标
            # 将输出 reshape 回 (N, 98, 2) 的格式，以便计算欧氏距离
            landmark_pred = landmark_pred.reshape(landmark_pred.shape[0], -1, 2).cpu().numpy()
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()
            
            # 使用 utils 里的函数计算 NME（通常是关键点距离除以瞳孔间距或外眼角间距）
            nme_temp = compute_nme(landmark_pred, landmark_gt)
            for item in nme_temp:
                nme_list.append(item)

    return np.mean(losses), np.mean(nme_list)


def main():
    cfg = get_config()

    SEED = cfg.SEED
    np.random.seed(SEED)
    # 管辖范围：Numpy 库，应用于数据预处理、数据增强等使用了 Numpy 随机数的操作
    random.seed(SEED)
    # 管辖范围：Python 原生的 random 库，应用于所有 Python 原生的随机数操作
    torch.manual_seed(SEED)
    # 管辖范围：PyTorch CPU，应用于所有在 CPU 上进行的随机数操作，例如神经网络权重的初始化、Dropout层在 CPU 上的随机丢弃、CPU 张量的生成
    torch.cuda.manual_seed(SEED)
    # 管辖范围：PyTorch GPU，应用于所有在当前 GPU 设备上进行的随机数操作

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # deterministic = True: 告诉 NVIDIA 的 CUDNN 库（负责卷积加速），在进行卷积运算时，必须使用确定的算法。因为为了追求极致速度，CUDNN 有时会自动选择一些非确定性的算法，导致即使种子固定了，跑出来的结果还有微小差异。开启此项会牺牲一点点训练速度，确保结果严格一致。

    warnings.filterwarnings("ignore")
    # 屏蔽（忽略）所有的警告信息，不打印到控制台，防止干扰训练进度条
    set_logger(cfg.LOGGER_PATH)
    # 设置日志管理器，由 File Hander 和 Stream Handler 组成，分别处理日志文件和控制台日志

    write_cfg(logging, cfg)
    # 将本次训练的配置信息写入info级别的日志，方便日后查阅

    torch.cuda.set_device(cfg.GPU_ID)
    # 指定要使用哪一块 GPU，如果有多块 GPU 的话

    main_worker(cfg)


def main_worker(cfg):
    # ======= LOADING DATA (数据加载) ======= #
    logging.warning('=======>>>>>>> Loading Training and Validation Data')
    # 这一个信息也会被之前 set_logger 函数配置好的 logger 处理。因为 logging.getLogger() 获取的是全局默认的根日志记录器（Root Logger），它是整个日志系统的入口 
    TRAIN_DATA_PATH = cfg.TRAIN_DATA_PATH
    VAL_DATA_PATH = cfg.VAL_DATA_PATH
    TRANSFORM = cfg.TRANSFORM

    # 初始化训练集 Datasets (负责读取单个样本)
    train_dataset = WLFWDatasets(TRAIN_DATA_PATH, TRANSFORM)
    # 初始化训练集 DataLoader (负责批量加载、打乱、多进程读取)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.TRAIN_BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,  # 稍微改大一点，充分利用你的至强 CPU
        drop_last=True,  # 建议改为True，丢弃最后几个凑不够Batch的数据，防止维度错误
        pin_memory=True, # <--- 关键！开启后数据传输速度翻倍
        prefetch_factor=4 # <--- 关键！让CPU提前准备4个Batch的数据
    )
    
    val_dataset = WLFWDatasets(VAL_DATA_PATH, TRANSFORM)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.VAL_BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True  # <--- 验证集也开启
    )

    # ======= MODEL (模型构建) ======= #
    MODEL_DICT = {'PFLD': PFLD,
                  'PFLD_GhostNet': PFLD_GhostNet,
                  'PFLD_GhostNet_Slim': PFLD_GhostNet_Slim,
                  'PFLD_GhostOne': PFLD_GhostOne,
                  }
    MODEL_TYPE = cfg.MODEL_TYPE
    WIDTH_FACTOR = cfg.WIDTH_FACTOR
    INPUT_SIZE = cfg.INPUT_SIZE
    LANDMARK_NUMBER = cfg.LANDMARK_NUMBER
    
    # 根据配置初始化对应的模型结构，并移动到指定设备 (GPU/CPU)
    model = MODEL_DICT[MODEL_TYPE](WIDTH_FACTOR, INPUT_SIZE[0], LANDMARK_NUMBER).to(cfg.DEVICE)

    # !!! 新增下面这一行 !!!
    # 利用 PyTorch 2.x 的图编译功能加速模型
    # mode='reduce-overhead' 适合小模型（如 MobileOne/GhostOne）
    model = torch.compile(model, mode='reduce-overhead')

    # model.apply(init_weights)
    if cfg.RESUME:
        if os.path.isfile(cfg.RESUME_MODEL_PATH):
            model.load_state_dict(torch.load(cfg.RESUME_MODEL_PATH))
        else:
            logging.warning("MODEL: No Checkpoint Found at '{}".format(cfg.RESUME_MODEL_PATH))
    logging.warning('=======>>>>>>> {} Model Generated'.format(MODEL_TYPE))

    # ======= LOSS (损失函数) ======= #
    # 使用自定义的关键点损失函数 (通常包含 WingLoss, SmoothL1 等)
    loss_fn = LandmarkLoss(LANDMARK_NUMBER)
    logging.warning('=======>>>>>>> Loss Function Generated')

    # ======= OPTIMIZER (优化器) ======= #
    # Adam 优化器：一种自适应学习率的优化算法，收敛速度快
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}],
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY)
    logging.warning('=======>>>>>>> Optimizer Generated')

    # ======= SCHEDULER (学习率调度器) ======= #
    # MultiStepLR：在指定的 epoch (milestones) 将学习率衰减 gamma 倍 (0.1)
    # 用于在训练后期微调模型，使其更稳定地收敛
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MILESTONES, gamma=0.1)
    logging.warning('=======>>>>>>> Scheduler Generated' + '\n')

    # ======= TENSORBOARDX WRITER (可视化记录器) ======= #
    writer = SummaryWriter(cfg.LOG_PATH)
    # 这行代码负责初始化 TensorBoard 的记录器。
    # TensorBoard 是 TensorFlow 自带的可视化工具（但也完美支持 PyTorch），用于在浏览器中实时监控训练进度（Loss曲线、Accuracy曲线、学习率变化、权重直方图等）。
    # SummaryWriter: 这是 PyTorch 提供的一个类，专门用来向硬盘写入 TensorBoard 格式的日志文件（events file）。
    # cfg.LOG_PATH: 指定这些日志文件保存的文件夹路径。
    # 后续用法: 在训练循环中，你会看到 writer.add_scalar('Train_Loss', train_loss, epoch) 这样的代码。这就是在告诉 writer：“把当前这一轮的 loss 值记录下来”。

    dummy_input = torch.rand(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(cfg.DEVICE)
    # 服务器新版 PyTorch 已经支持直接使用 SummaryWriter 记录计算图
    # writer.add_graph(model, (dummy_input,)) 

    best_nme = float('inf')

    # 初始化 Rich 进度条，用于在终端显示美观的训练进度
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TimeRemainingColumn(),
                  TimeElapsedColumn()) as progress:
        # TextColumn("[progress.description]..."): 显示任务描述文本（例如左侧的 "Epoch Process :"）。
        # BarColumn(): 显示那个动态增长的进度条图形（例如 [██████████] ）。
        # TextColumn("[progress.percentage]..."): 显示百分比数值（例如 50%）。
        # TimeRemainingColumn(): 显示剩余时间（根据当前速度估算还需要跑多久，非常实用）。
        # TimeElapsedColumn(): 显示已用时间。

        # 定义主任务：Epoch 进度
        epoch_task = progress.add_task(description="[red]Epoch Process :", total=cfg.EPOCHES)
        # 定义子任务：每个 Epoch 内的 Batch 进度
        batch_task = progress.add_task(description="", total=len(train_dataloader))
        # 这里的 len(train_dataloader) 会返回训练集中有多少个 Batch（总样本数除以 Batch Size 向上取整）
        # 样本数就是 len(train_dataset)，根据WFLWDatasets类的魔术方法给出
        # 定义子任务：测试进度
        test_task = progress.add_task(description="[cyan]Test :")

        for epoch in range(1, cfg.EPOCHES + 1):
            progress.advance(epoch_task, advance=1) 
            # 让进度条的 Epoch 任务前进一步
    
            progress.reset(batch_task)
            # 把指定任务的进度条归零，就像秒表清零一样。
            progress.reset(test_task)
            progress.update(batch_task, description="[green]Epoch {} :".format(epoch), total=len(train_dataloader))
            # 修改进度条的属性，比如修改左边的文字描述（从 "Epoch 1:" 改成 "Epoch 2:"），或者重新设置总长度（虽然长度通常不变，但属于标准写法）。
            
            # 1. 执行训练
            train_loss = train(model, train_dataloader, loss_fn, optimizer, cfg, progress, batch_task)
            # 2. 执行验证
            # VAL 会返回两个指标，一个是 Loss，另一个是 NME指标
            val_loss, val_nme = validate(model, val_dataloader, loss_fn, cfg, progress, test_task)
            # 3. 更新学习率
            scheduler.step()

            # 4. 保存最佳模型 (Best Checkpoint)（根据 NME 评判）
            if val_nme < best_nme:
                best_nme = val_nme
                save_checkpoint(cfg, model, extra='best')
                logging.info('Save best model')
            
            # 5. 保存当前 Epoch 模型 (用于断点续训)
            save_checkpoint(cfg, model, epoch)

            # 6. 记录日志到 TensorBoard
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Train_Loss', train_loss, epoch)
            writer.add_scalar('Val_Loss', val_loss, epoch)
            writer.add_scalar('Val_NME', val_nme, epoch)

            # 7. 打印日志到控制台/文件
            logging.info('Train_Loss: {}'.format(train_loss))
            logging.info('Val_Loss: {}'.format(val_loss))
            logging.info('Val_NME: {}'.format(val_nme) + '\n')

    # 训练结束后保存最终模型
    save_checkpoint(cfg, model, extra='final')
    # 这一步是保存.pth文件
    # .pth文件的组成


if __name__ == "__main__":
    main()
