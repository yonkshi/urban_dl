from eval_unet_xview2 import inference_loop

import sys
import os
from os import path, listdir
from os.path import join, isfile
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
from unet.utils import Xview2Detectron2Dataset
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
    cfg.merge_from_file(f'/home/yonk/urban_dl/urban_dl/configs/{MODEL_NAME}.yaml')
else:
    cfg.merge_from_file(f'/Midgard/home/pshi/urban_dl/configs/{MODEL_NAME}.yaml')
cfg.OUTPUT_DIR = path.join(cfg.OUTPUT_BASE_DIR, MODEL_NAME)
net = UNet(cfg)

# load state dict
full_model_path = os.path.join(cfg.OUTPUT_DIR, CHECKPOINT_NAME)
net.load_state_dict(torch.load(full_model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if TRAIN_TYPE == 'train':
    dset_source = cfg.DATASETS.TRAIN[0]
else:
    dset_source = cfg.DATASETS.TEST[0]

with open(dset_source + '/labels.json') as f:
    ds = json.load(f)
dataset_json = ds

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
    dataloader = torch_data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKER, shuffle = False, drop_last=False,)

    dataset_length = np.minimum(len(dataset), max_samples)
    with torch.no_grad():
        for step, (imgs, y_label, sample_name, index) in enumerate(dataloader):

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

            callback(imgs, y_label, y_pred, sample_name, index)


            if (max_samples is not None) and step >= max_samples:
                break



# Per image  ===========

print('================= Running ablation per image ===============', flush=True)

trfm = []
if cfg.AUGMENTATION.RESIZE: trfm.append(Resize(scale=cfg.AUGMENTATION.RESIZE_RATIO, resize_label=False))
trfm.append(Npy2Torch())
trfm = transforms.Compose(trfm)

dataset = Xview2Detectron2Dataset(dset_source, include_index=True, transform=trfm)
results_table = []


def compute_sample(x, Y_true, Y_pred, img_filenames, indices):
    # interp image if scaling was originally enabled
    if cfg.AUGMENTATION.RESIZE:
        upscale_ratio = 1 / cfg.AUGMENTATION.RESIZE_RATIO
        Y_pred = torch.nn.functional.interpolate(Y_pred,
                                                 scale_factor=upscale_ratio,
                                                 mode='bilinear')
    # expand batch
    Y_pred = Y_pred.squeeze(1) > THRESHOLD  # remove empty channel and activate Y_pred
    Y_true = Y_true.type(torch.bool)

    hw_dims = (-1, -2)
    # Compute all TP, TN, FP, FP, FN at once
    bTP = (Y_true & Y_pred).sum(dim=hw_dims).cpu()
    bTN = (~Y_true & ~Y_pred).sum(dim=hw_dims).cpu()
    bFP = (~Y_true & Y_pred).sum(dim=hw_dims).cpu()
    bFN = (Y_true & ~Y_pred).sum(dim=hw_dims).cpu()
    bAreas = Y_true.sum(dim=hw_dims).cpu()

    # All false positive pixels
    bFP_pixels = (~Y_true & Y_pred).cpu().numpy()
    # SDT = Signed Distance Transform
    bSDT_FP_maps = (~Y_true).cpu().numpy() # All negative (both false pos + true neg) pixels to 1, all positive pixels to 0

    # All false negative pixels
    bFN_pixels = (~Y_true & Y_pred).cpu().numpy()
    bSDT_FN_maps = Y_true.cpu().numpy()

    # Iterate through batch
    for fn_pixels, sdt_fn_map, fp_pixels, sdt_fp_map, tp, tn, fp, fn, total_area, img_filename, index in zip(bFN_pixels, bSDT_FN_maps, bFP_pixels, bSDT_FP_maps, bTP, bTN, bFP, bFN, bAreas, img_filenames, indices):
        tp = tp.item()
        tn = tn.item()
        fp = fp.item()
        fn = fn.item()

        # print(annotation_set.keys(), flush=True)
        total_area = total_area.item()
        density = len(dataset_json[index]['annotations'])

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

        # Distance transform, distance from all negative pixels to positive pixels
        sdt = distance_transform_edt(sdt_fp_map)
        fp_distances = sdt[fp_pixels]

        distance_intervals = 2 ** np.arange(1, 11)
        for interval in distance_intervals:
            interval_mask = fp_distances <= interval
            fp_distances = fp_distances[~interval_mask] # next_interval
            result[f'fp_sdt<={interval}'] = interval_mask.sum()

        # Distance transform, distance from all negative pixels to positive pixels
        sdt_fn = distance_transform_edt(sdt_fn_map)
        fn_distances = sdt_fn[fn_pixels]
        for interval in distance_intervals:
            interval_mask = fn_distances <= interval
            fn_distances = fn_distances[~interval_mask] # next_interval
            result[f'fn_sdt<={interval}'] = interval_mask.sum()

        results_table.append(result)


inference_loop2(net, cfg, device, compute_sample,
                dataset=dataset)
results = pd.DataFrame(results_table)
storage_path = path.join(cfg.OUTPUT_DIR, 'ablation',)
os.makedirs(storage_path, exist_ok=True)
results.to_pickle(path.join(storage_path, f'per_image_result_{TRAIN_TYPE}.pkl'))


# ===========
# Per building
# ===========
print('================= Running ablation per building ===============', flush=True)

dataset = Xview2Detectron2Dataset(dset_source, include_index=True, transform=trfm)
results_table = []


def compute_sample(x, Y_true, Y_pred, img_filenames, indices):

    # interp image if scaling was originally enabled
    if cfg.AUGMENTATION.RESIZE:
        upscale_ratio = 1 / cfg.AUGMENTATION.RESIZE_RATIO
        Y_pred = torch.nn.functional.interpolate(Y_pred,
                                                 scale_factor=upscale_ratio,
                                                 mode='bilinear')

    # expand batch
    Y_pred = Y_pred.squeeze(1) > THRESHOLD
    Y_true = Y_true.type(torch.bool)
    # Compute all TP, TN, FP, FP, FN at once
    bTP = (Y_true & Y_pred)
    bTN = (~Y_true & ~Y_pred)
    bFP = (Y_true & ~Y_pred)
    bFN = (~Y_true & Y_pred)

    # Iterate through batch
    for y_pred, TP, TN, FP, FN, img_filename, index in zip(Y_pred, bTP, bTN, bFP, bFN, img_filenames, indices):
        y_pred = y_pred.cpu().numpy()
        num_buildings = len(dataset_json[index]['annotations'])


        # iterate through buildings
        for idx, anno in enumerate(dataset_json[index]['annotations']):
            x1, y1, x2, y2 = np.array(anno['bbox']).astype(np.int32)

            crop_range = (slice(y1 - 2, y2 + 2), slice(x1 - 2, x2 + 2))  # equiv: [y1-2:y2+2, x1-2:x2+2]

            tp = TP[crop_range].sum().item()
            tn = TN[crop_range].sum().item()
            fp = FP[crop_range].sum().item()
            fn = FN[crop_range].sum().item()

            height = y2 - y1
            width = x2 - x1

            # Real Per building A

            seg = np.array(anno['segmentation'], dtype=np.int32)
            seg_xy = seg.reshape(-1, 2)
            building_poly = np.zeros((1024, 1024), dtype=np.uint8)
            cv2.fillConvexPoly(building_poly, seg_xy , 1)
            y_true_real = building_poly.astype(np.bool)

            real_area = y_true_real.sum()
            real_tp = (y_pred & y_true_real).sum()
            real_fn = real_area - real_tp

            result = {
                'index': index.item(),
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,

                'image_name': img_filename,
                'height': height,
                'width': width,
                'area': height * width,

                'real_TP': real_tp,
                'real_FN': real_fn,
                'real_area': real_area,

            }
            results_table.append(result)


inference_loop2(net, cfg, device, compute_sample,
                dataset=dataset)

results = pd.DataFrame(results_table)
storage_path = path.join(cfg.OUTPUT_DIR, 'ablation',)
os.makedirs(storage_path, exist_ok=True)
results.to_pickle(path.join(storage_path, f'per_building_result_{TRAIN_TYPE}.pkl'))