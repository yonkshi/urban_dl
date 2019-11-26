import sys
import os
from os import path, listdir
from os.path import join, isfile

from argparse import ArgumentParser

import numpy as np
import torch
import wandb
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils import data as torch_data
from tabulate import tabulate
from debug_tools import __benchmark_init, benchmark
from unet import UNet
from unet.utils import Xview2Detectron2Dataset
from experiment_manager.metrics import roc_score, f1_score, MultiThresholdMetric
from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config
from experiment_manager.utils import to_numpy
from sklearn.metrics import roc_auc_score, average_precision_score
# import hp

def final_model_evaluation_runner(net, cfg,):
    '''
    Runner that only concerns with only a single model,
    :return:
    '''

    # Setup
    F1_THRESH = torch.linspace(0, 1, 100).to(device)
    f1_set = []
    fpr_set = [] # False positive rate
    tpr_set = []

    def evaluate(y_true, y_pred):

        measurer = MultiThresholdMetric(y_true, y_pred, F1_THRESH)
        # Compute F1 per batch
        f1 = measurer.compute_f1()
        f1_set.append(f1)

        # Compute ROC per batch
        fpr, tpr = measurer.compute_roc_curve()
        fpr_set.append(fpr)
        tpr_set.append(tpr)

    inference_loop(net, cfg, device, evaluate)

    # ===
    # Collect for summary

    # Mean F1 score
    mF1 = torch.cat(f1_set, dim = 0)
    mF1 = to_numpy(torch.mean(mF1, dim=0))



    # ROC curve
    # ROC metrics are already in numpy
    mFPR = torch.cat(fpr_set, dim=0)
    mFPR = to_numpy(torch.mean(mFPR, dim=1))

    mTPR = torch.cat(tpr_set, dim=0)
    mTPR = to_numpy(torch.mean(mTPR, dim=1))

    import matplotlib.pyplot as plt
    plt.plot(mFPR, mTPR)
    plt.ylabel('true_positive')
    plt.xlabel('false_positive')
    plt.title('ROC curve')
    wandb.log({'roc': plt})

    # Log to wandb
    for thresh, (f1_thresh, true_pos, false_pos) in enumerate(zip(mF1, mTPR, mFPR)):
        wandb.log({'f1': f1_thresh, 'thresholds':thresh})
        wandb.log({'True positive rate': true_pos, 'False Positive rate': false_pos})

    print('done')

def model_checkpoints_eval_runner(net, cfg):
    # TODO Limit to how much inference loop can sample from training set (So we don't end up training the entire training set)

    checkpoint_files = list_and_sort_checkpoint_files()

    for cp_num,cp_file in checkpoint_files:
        full_model_path = os.path.join(cfg.OUTPUT_DIR, cp_file)
        net.load_state_dict(torch.load(full_model_path))

        # TRAINING EVALUATION
        maxF1, mAUC, mAP = model_eval(net, cfg, device, run_type='TRAIN')
        wandb.log({'training_set max F1': maxF1,
                   'training_set mean AUC score': mAUC,
                   'training_set mean Average Precision': mAP,
                   'step': cp_num
                   })


        # TEST SET EVALUATION
        maxF1, mAUC, mAP = model_eval(net, cfg, device, run_type='TEST')
        wandb.log({'test_set max F1': maxF1,
                   'test_set mean AUC score': mAUC,
                   'test_set mean Average Precision': mAP,
                   'step': cp_num
                   })




def model_eval(net, cfg, device, run_type='TEST'):
    '''
    Runner that is concerned with training changes
    :param run_type: 'train' or 'eval'
    :return:
    '''

    F1_THRESH = torch.linspace(0, 1, 100).to(device)
    auc_set = []
    f1_set = []
    ap_set = []
    def evaluate(y_true, y_pred):
        # FIXME Naively assuming batch_size = 1

        measurer = MultiThresholdMetric(y_true, y_pred, F1_THRESH)
        y_true_np = to_numpy(y_true.flatten())
        y_pred_np = to_numpy(y_pred.flatten())

        # Max F1
        f1 = measurer.compute_f1()
        f1_set.append(f1)


        if y_true_np.max() == 1: # Ignore empty images
            # Area under curve
            auc_score = roc_auc_score(y_true_np, y_pred_np)
            auc_set.append(auc_score)

            # Average Precision
            ap_score = average_precision_score(y_true_np, y_pred_np)
            ap_set.append(ap_score)
    if run_type == 'TRAIN':
        inference_loop(net, cfg, device, evaluate, 'TRAIN', max_samples=1000)
    elif run_type == 'TEST':
        inference_loop(net, cfg, device, evaluate)

    # Summary gathering ===

    # Max of the mean F1 score
    mF1 = torch.cat(f1_set, dim = 0)
    maxF1 = mF1.mean(dim=0).max()
    mAUC = np.mean(auc_set)
    mAP = np.mean(ap_set)
    return maxF1, mAUC, mAP

def inference_loop(net, cfg, device, callback = None, run_type = 'TEST', max_samples = None,
              ):

    net.to(device)
    net.eval()

    # reset the generators

    dset_source = cfg.DATASETS.TEST[0] if run_type == 'TEST' else cfg.DATASETS.TRAIN[0]
    dataset = Xview2Detectron2Dataset(dset_source, 0)
    dataloader = torch_data.DataLoader(dataset,
                                       batch_size=1,
                                       num_workers=cfg.DATALOADER.NUM_WORKER,
                                       shuffle = cfg.DATALOADER.SHUFFLE,
                                       drop_last=True,
                                       )

    dataset_length = len(dataset)
    roc_set = []
    with torch.no_grad():
        for step, (imgs, y_label, sample_name) in enumerate(dataloader):
            imgs = imgs.to(device)
            y_label = y_label.to(device)

            y_pred = net(imgs)
            y_pred = torch.sigmoid(y_pred)

            if step % 100 == 0 or step == dataset_length-1:
                print(f'Processed {step+1}/{dataset_length}')

            if callback:
                callback(y_label, y_pred)

            if (max_samples is not None) and step >= max_samples:
                break

def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/unet/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file

    if args.log_dir: # Override Output dir
        cfg.OUTPUT_DIR = path.join(args.log_dir, args.config_file)
    else:
        cfg.OUTPUT_DIR = path.join(cfg.OUTPUT_BASE_DIR, args.config_file)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.data_dir:
        cfg.DATASETS.TRAIN = (args.data_dir,)
    return cfg

def list_and_sort_checkpoint_files():
    '''
    Iterate through the designated log directory and load all
    :return:
    '''
    file_list = [(int(f.split('_')[1].split('.')[0]),f)  # e.g. (123, 'cp_123.pkl')
                 for f in listdir(cfg.OUTPUT_DIR)
                 if isfile(join(cfg.OUTPUT_DIR, f))
                 and f.endswith('.pkl')
                 and len(f.split('_')) == 2 # ignore saved models that aren't checkpoints
                 ]

    file_list.sort(key=lambda fname: fname[0])
    return file_list
if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg = setup(args)
    wandb.init(
        name=cfg.NAME,
        project='urban_dl',
        tags=['inference']
    )

    # torch.set_default_dtype(torch.float16)f
    out_channels = 1 if cfg.MODEL.BINARY_CLASSIFICATION else cfg.MODEL.OUT_CHANNELS
    net = UNet(n_channels=cfg.MODEL.IN_CHANNELS, n_classes=out_channels)
    print('ready to run')
    if args.resume_from: # TODO Remove this
        full_model_path = os.path.join(cfg.OUTPUT_DIR, args.resume_from)
        net.load_state_dict(torch.load(full_model_path))
        print('Model loaded from {}'.format(full_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if args.eval_training:
            model_checkpoints_eval_runner(net, cfg)
        else:
            final_model_evaluation_runner(net, cfg, device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


