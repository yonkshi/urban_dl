from eval_unet_xview2 import inference_loop
from os import path, listdir
import json
import argparse

from argparse import ArgumentParser

import numpy as np
import torch
import pandas as pd
import cv2
from scipy.ndimage.morphology import distance_transform_edt
from torchvision import transforms, utils

from torch.utils import data as torch_data
from unet import UNet
from unet.dataloader import Xview2Detectron2Dataset
from unet.augmentations import *
from experiment_manager.config import new_config


parser = argparse.ArgumentParser(description="Experiment Args")
parser.add_argument('-c', "--config-file", dest='config_file', required=True, help="path to config file")
parser.add_argument('-p', '--checkpoint', dest='checkpoint_name', type=str,required=True, help='dataset directory')
parser.add_argument('-t', '--threshold', dest='threshold', type=float,required=True, help='dataset directory')
parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=5, help='output directory')
parser.add_argument( '-d', "--debug", dest='debug', action="store_true", help="whether to attempt to resume from the checkpoint directory", )
args = parser.parse_known_args()[0]

MODEL_NAME = args.config_file
CHECKPOINT_NAME = args.checkpoint_name
THRESHOLD = args.threshold
BATCH_SIZE = args.batch_size
TRAIN_TYPE = 'test'

torch.manual_seed(0)

cfg = new_config()
if args.debug:
    cfg.merge_from_file(f'/home/yonk/urban_dl/urban_dl/configs/damage_detection/{MODEL_NAME}.yaml')
else:
    cfg.merge_from_file(f'/Midgard/home/pshi/urban_dl/configs/damage_detection/{MODEL_NAME}.yaml')

cfg.OUTPUT_DIR = path.join(cfg.OUTPUT_BASE_DIR, MODEL_NAME)
net = UNet(cfg)
print('==========  THRESHOLD', cfg.THRESH)

# load state dict
full_model_path = os.path.join(cfg.OUTPUT_DIR, CHECKPOINT_NAME)
net.load_state_dict(torch.load(full_model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if TRAIN_TYPE == 'train':
    dset_source = cfg.DATASETS.TRAIN[0]
else:
    dset_source = cfg.DATASETS.TEST[0]


def compute_sample(x, Y_true, Y_pred, img_filenames, indices):
    # interp image if scaling was originally enabled
    if cfg.AUGMENTATION.RESIZE:
        upscale_ratio = 1 / cfg.AUGMENTATION.RESIZE_RATIO
        Y_pred = torch.nn.functional.interpolate(Y_pred,
                                                 scale_factor=upscale_ratio,
                                                 mode='bilinear')
    # expand batch





inference_loop(net, cfg, device, compute_sample,
                dataset=dataset)
