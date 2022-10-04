import time
import copy
import datetime
import os
import sys

from tqdm import tqdm

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.solver.loss import get_segmentation_loss
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params, get_color_pallete
from segmentron.config import cfg
from tabulate import tabulate
from IPython import embed
from PIL import Image
try:
    import apex
except:
    print('apex is not installed')

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.use_fp16 = cfg.TRAIN.APEX

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform,
                       'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}

        data_kwargs_testval = {'transform': input_transform,
                       'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TEST.CROP_SIZE}

        mymap = {True:[], False:[]}
        ip_to_ele = {}
        train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode='testval', **data_kwargs_testval)
        test_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='test', mode='testval', **data_kwargs_testval)
        for is_thing, name, elements in tqdm(train_dataset):
            mymap[is_thing]+=[name]
            ip_to_ele[name]=elements
        print(ip_to_ele)
        print(mymap)


if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train' if not args.test else 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)

    # create a trainer and start train
    trainer = Trainer(args)
    # if args.test:
    #     assert 'pth' in cfg.TEST.TEST_MODEL_PATH, 'please provide test model pth!'
    #     logging.info('test model......')
    #     trainer.test(args.vis)
    # else:
    #     trainer.train()
    #     trainer.test()