from torchvision import transforms as trans
from easydict import EasyDict as edict
from utils.utils import get_time
import os
import torch


def get_config():
    cfg = edict()
    # 用 easydict 替换原生字典，使得可以用 cfg.KEY 的形式替换 cfg['KEY']
    cfg.SEED = 2023
    cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.GPU_ID = 0
    # 指定要使用哪一块 GPU，如果有多块 GPU 的话

    cfg.TRANSFORM = trans.Compose([trans.ToTensor(),
                                   trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    cfg.MODEL_TYPE = 'PFLD_GhostOne'  # [PFLD, PFLD_GhostNet, PFLD_GhostNet_Slim, PFLD_GhostOne]
    cfg.INPUT_SIZE = [112, 112]
    cfg.WIDTH_FACTOR = 1
    cfg.LANDMARK_NUMBER = 98

    # Updated for RTX 5090 32GB
    cfg.TRAIN_BATCH_SIZE = 400
    cfg.VAL_BATCH_SIZE = 128

    cfg.TRAIN_DATA_PATH = './data/train_data_repeat80/list.txt'
    cfg.VAL_DATA_PATH = './data/test_data_repeat80/list.txt'

    cfg.EPOCHES = 80
    cfg.LR = 1e-4
    # 学习率
    cfg.WEIGHT_DECAY =1e-6
    cfg.NUM_WORKERS = 24
    cfg.MILESTONES = [55, 65, 75]
    # 学习率调整节点，在第55, 65, 75个 epoch 时学习率会自动缩小

    cfg.RESUME = False
    # 是否恢复训练（断点续训）
    if cfg.RESUME:
        cfg.RESUME_MODEL_PATH = ''
        # 如果要恢复训练，指定模型路径

    create_time = get_time()
    cfg.MODEL_PATH = './checkpoint/models/{}_{}_{}_{}/'.format(cfg.MODEL_TYPE, cfg.WIDTH_FACTOR, cfg.INPUT_SIZE[0], create_time)
    cfg.LOG_PATH = './checkpoint/log/{}_{}_{}_{}/'.format(cfg.MODEL_TYPE, cfg.WIDTH_FACTOR, cfg.INPUT_SIZE[0], create_time)
    cfg.LOGGER_PATH = os.path.join(cfg.MODEL_PATH, "train.log")
    if not os.path.exists(cfg.MODEL_PATH):
        os.makedirs(cfg.MODEL_PATH)

    return cfg
