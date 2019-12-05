from eval_unet_xview2 import inference_loop

import sys
import os
from os import path, listdir
from os.path import join, isfile
import json

from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd


from torch.utils import data as torch_data
from tabulate import tabulate
from debug_tools import __benchmark_init, benchmark
from unet import UNet
from unet.utils import Xview2Detectron2Dataset
from experiment_manager.metrics import roc_score, f1_score, MultiThresholdMetric
from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config
from experiment_manager.utils import to_numpy
from experiment_manager.dataset import SimpleInferenceDataset
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

MODEL_NAME = 'unet_deeper_diceloss'
CHECKPOINT_NAME = 'cp_100000.pkl'
THRESHOLD = 0.8
BATCH_SIZE = 4
plt.rcParams['figure.figsize'] = [15, 30]
torch.manual_seed(0)

cfg = new_config()
cfg.merge_from_file(f'/Midgard/home/pshi/urban_dl/configs/unet/{MODEL_NAME}.yaml')
cfg.OUTPUT_DIR = path.join(cfg.OUTPUT_BASE_DIR, MODEL_NAME)
net = UNet(cfg)

# load state dict
full_model_path = os.path.join(cfg.OUTPUT_DIR, CHECKPOINT_NAME)
net.load_state_dict(torch.load(full_model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



with open(cfg.DATASETS.TRAIN[0] + '/labels.json') as f:
    ds = json.load(f)
dataset_train = ds
with open(cfg.DATASETS.TEST[0] + '/labels.json') as f:
    ds = json.load(f)
dataset_test = ds
def inference_loop2(net, cfg, device,
                   callback = None,
                   run_type = 'TEST',
                   max_samples = 999999999,
                   dataset = None
              ):

    net.to(device)
    net.eval()

    # reset the generators
    if dataset is None:
        dataset = Xview2Detectron2Dataset(dset_source, 0, cfg)
    dataloader = torch_data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle = False, drop_last=False,)

    dlen = len(dataset)
    dataset_length = np.minimum(len(dataset), max_samples)
    with torch.no_grad():
        for step, (imgs, y_label, sample_name, raw_label) in enumerate(dataloader):
            imgs = imgs.to(device)
            y_label = y_label.to(device)

            y_pred = net(imgs)

            if step % 10 == 0 or step == dataset_length-1:
                print(f'Processed {step+1}/{dataset_length}', f', max cuda usage: {torch.cuda.max_memory_allocated() / 1e6 :.2f} MB', flush=True)

            if cfg.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
                # In Two class Cross entropy mode, positive classes are in Channel #2
                y_pred = torch.softmax(y_pred, dim=1)
                y_pred = y_pred[:,1 ,...]
                y_pred = y_pred[:, None, ...]
            else:
                y_pred = torch.sigmoid(y_pred)

            callback(imgs, y_label, y_pred, sample_name, raw_label)


            if (max_samples is not None) and step >= max_samples:
                break


dset_source = cfg.DATASETS.TRAIN[0]
dataset = Xview2Detectron2Dataset(dset_source, cfg, random_crop=False, include_index=True)  # TODO return raw label
results_table = []

# Per image
def compute_sample(x, Y_true, Y_pred, img_filenames, indices):
    # expand batch
    Y_pred = Y_pred.squeeze() > THRESHOLD  # remove empty channel and activate Y_pred
    Y_true = Y_true.type(torch.bool)

    hw_dims = (-1, -2)
    # Compute all TP, TN, FP, FP, FN at once
    bTP = (Y_true & Y_pred).sum(dim=hw_dims)
    bTN = (~Y_true & ~Y_pred).sum(dim=hw_dims)
    bFP = (Y_true & ~Y_pred).sum(dim=hw_dims)
    bFN = (~Y_true & Y_pred).sum(dim=hw_dims)
    bAreas = Y_true.sum(dim=hw_dims)

    # Iterate through batch
    for tp, tn, fp, fn, total_area, img_filename, index in zip(bTP, bTN, bFP, bFN, bAreas, img_filenames, indices):
        tp = tp.sum().item()
        tn = tn.sum().item()
        fp = fp.sum().item()
        fn = fn.sum().item()

        # print(annotation_set.keys(), flush=True)
        total_area = total_area.item()
        density = len(dataset_train[index]['annotations'])

        result = {
            'index': index.item(),
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,

            'image_name': img_filename,
            'density': density,
            'total_area': total_area,
        }
        results_table.append(result)


inference_loop2(net, cfg, device, compute_sample,
                dataset=dataset)
results = pd.DataFrame(results_table)
results.to_pickle('ablation_per_image_result_train.pkl')




# Per building

dset_source = cfg.DATASETS.TRAIN[0]
dataset = Xview2Detectron2Dataset(dset_source, cfg, random_crop=False, include_index=True)  # TODO return raw label
results_table = []


def compute_sample(x, Y_true, Y_pred, img_filenames, indices):
    # expand batch
    Y_pred = Y_pred.squeeze() > THRESHOLD
    Y_true = Y_true.type(torch.bool)
    # Compute all TP, TN, FP, FP, FN at once
    bTP = (Y_true & Y_pred)
    bTN = (~Y_true & ~Y_pred)
    bFP = (Y_true & ~Y_pred)
    bFN = (~Y_true & Y_pred)

    # Iterate through batch
    for TP, TN, FP, FN, img_filename, index in zip(bTP, bTN, bFP, bFN, img_filenames, indices):

        # iterate through buildings
        for anno in dataset_train[index]['annotations']:
            x1, y1, x2, y2 = np.array(anno['bbox']).astype(np.int32)

            crop_range = (slice(y1 - 2, y2 + 2), slice(x1 - 2, x2 + 2))  # equiv: [y1-2:y2+2, x1-2:x2+2]

            tp = TP[crop_range].sum().item()
            tn = TN[crop_range].sum().item()
            fp = FP[crop_range].sum().item()
            fn = FN[crop_range].sum().item()

            height = y2 - y1
            width = x2 - x1
            result = {
                'index': index.item(),
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,

                'image_name': img_filename,
                'height': height,
                'width': width,
                'area': height * width

            }
            results_table.append(result)


inference_loop2(net, cfg, device, compute_sample,
                dataset=dataset)

results = pd.DataFrame(results_table)
results.to_pickle('ablation_per_building_result_train.pkl')