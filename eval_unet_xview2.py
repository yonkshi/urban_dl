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
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
# import hp

def final_model_evaluation_runner(net, cfg):
    '''
    Runner that only concerns with only a single model,
    :return:
    '''
    print('=== Evaluating final model ===')
    # Setup
    F1_THRESH = torch.linspace(0, 1, 100).to(device)
    f1_set = []
    fpr_set = [] # False positive rate
    tpr_set = []

    y_true_set = []
    y_pred_set = []
    measurer = MultiThresholdMetric(F1_THRESH)

    def evaluate(y_true, y_pred):
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        y_true_set.append(y_true.cpu())
        y_pred_set.append(y_pred.cpu())

        measurer.add_sample(y_true, y_pred)

    inference_loop(net, cfg, device, evaluate)

    # ===
    # Collect for summary

    y_true_set = torch.cat(y_true_set, dim = 0)
    y_pred_set = torch.cat(y_pred_set, dim=0)

    y_true_set, y_pred_set = downsample_dataset_for_eval(y_true_set, y_pred_set)

    y_true_np = to_numpy(y_true_set.flatten())
    y_pred_np = to_numpy(y_pred_set.flatten())

    # F1 score
    print('Computing F1 vs thresholds')
    f1 = measurer.compute_f1()

    print('computing ROC curve')
    # ROC curve
    fpr, tpr, thresh = roc_curve(y_true_np, y_pred_np)

    # Down sample roc curve or matplotlib won't like it
    num_ele = len(fpr)
    downsample_idx = np.linspace(0, num_ele, 1000, dtype=np.int32, endpoint=False)
    fpr_downsampled = fpr[downsample_idx]
    tpr_downsampled = tpr[downsample_idx]


    import matplotlib.pyplot as plt
    plt.plot(fpr_downsampled, tpr_downsampled)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.ylabel('true_positive rate')
    plt.xlabel('false_positive rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC curve')

    wandb.log({'roc': plt})

    # Log to wandb
    for thresh, f1_thresh in enumerate(f1):
        wandb.log({'f1': f1_thresh, 'thresholds':thresh})

    print('done')

def model_checkpoints_eval_runner(net, cfg):

    checkpoint_files = list_and_sort_checkpoint_files()

    for cp_num,cp_file in checkpoint_files:
        print(' ==== checkpoint', cp_file)
        full_model_path = os.path.join(cfg.OUTPUT_DIR, cp_file)
        net.load_state_dict(torch.load(full_model_path))

        # TRAINING EVALUATION
        maxF1, best_fpr, best_fnr,  mAUC, mAP = model_eval(net, cfg, device, run_type='TRAIN')
        wandb.log({'training_set max F1': maxF1,
                   'training_set AUC score': mAUC,
                   'training_set Average Precision': mAP,
                   'training_set false positive rate':best_fpr,
                   'training_set false negative rate': best_fnr,
                   'step': cp_num
                   })


        # TEST SET EVALUATION
        maxF1, best_fpr, best_fnr,   mAUC, mAP = model_eval(net, cfg, device, run_type='TEST')
        wandb.log({'test_set max F1': maxF1,
                   'test_set AUC score': mAUC,
                   'test_set Average Precision': mAP,
                   'test_set false positive rate': best_fpr,
                   'test_set false negative rate': best_fnr,
                   'step': cp_num
                   })

def model_eval(net, cfg, device, run_type='TEST', max_samples = 1000):
    '''
    Runner that is concerned with training changes
    :param run_type: 'train' or 'eval'
    :return:
    '''

    F1_THRESH = torch.linspace(0, 1, 100).to(device)
    y_true_set = []
    y_pred_set = []

    measurer = MultiThresholdMetric(F1_THRESH)

    def evaluate(y_true, y_pred):
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        y_true_set.append(y_true.cpu())
        y_pred_set.append(y_pred.cpu())

        measurer.add_sample(y_true, y_pred)

    if run_type == 'TRAIN':
        inference_loop(net, cfg, device, evaluate, 'TRAIN', max_samples = max_samples)
    elif run_type == 'TEST':
        inference_loop(net, cfg, device, evaluate, max_samples = max_samples)

    # Summary gathering ===

    print('Computing F1 score ', end=' ', flush=True)
    # Max of the mean F1 score

    # measurer = MultiThresholdMetric(y_true_set, y_pred_set, F1_THRESH)
    # Max F1


    f1 = measurer.compute_f1()
    fpr, fnr = measurer.compute_basic_metrics()
    maxF1 = f1.max()
    argmaxF1 = f1.argmax()
    best_fpr = fpr[argmaxF1]
    best_fnr = fnr[argmaxF1]
    print(maxF1.item(), flush=True)


    y_true_set = torch.cat(y_true_set, dim = 0)
    y_pred_set = torch.cat(y_pred_set, dim=0)

    y_true_set, y_pred_set = downsample_dataset_for_eval(y_true_set, y_pred_set)

    y_true_np = to_numpy(y_true_set.flatten())
    y_pred_np = to_numpy(y_pred_set.flatten())

    # Area under curve
    print('Computing AUC score  ', end='', flush=True)
    auc = roc_auc_score(y_true_np, y_pred_np)
    print(auc)

    # Average Precision
    print('Computing AP score ... ', end='', flush=True)
    ap = average_precision_score(y_true_np, y_pred_np)
    print(ap)

    return maxF1, best_fpr, best_fnr, auc, ap
def downsample_dataset_for_eval(y_true, y_pred):
    # Full dataset is too big to compute for the CPU, so we down sample it
    num_samples = y_pred.shape[0]
    downsample_size = 100
    if num_samples > downsample_size: # only downsample if data size is huge
        down_idx = np.linspace(0, num_samples, downsample_size, endpoint=False, dtype=np.int)
        y_pred = y_pred[down_idx]
        y_true = y_true[down_idx]

    return y_true, y_pred

def inference_loop(net, cfg, device, callback = None, run_type = 'TEST', max_samples = 999999999,
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

    dlen = len(dataset)
    dataset_length = np.minimum(len(dataset), max_samples)
    with torch.no_grad():
        for step, (imgs, y_label, sample_name) in enumerate(dataloader):
            imgs = imgs.to(device)
            y_label = y_label.to(device)

            y_pred = net(imgs)
            y_pred = torch.sigmoid(y_pred)

            if step % 100 == 0 or step == dataset_length-1:
                print(f'Processed {step+1}/{dataset_length}')

            if cfg.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
                # In Two class Cross entropy mode, positive classes are in Channel #2
                y_pred = y_pred[:,1 ,...]
                y_pred = y_pred[:, None, ...]

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
    print('ready to run 0')
    wandb.init(
        name=cfg.NAME,
        project='urban_dl',
        tags=['inference']
    )
    print('ready to run 1')
    # torch.set_default_dtype(torch.float16)f
    out_channels = 1 if cfg.MODEL.BINARY_CLASSIFICATION else cfg.MODEL.OUT_CHANNELS
    net = UNet(n_channels=cfg.MODEL.IN_CHANNELS, n_classes=out_channels)
    print('ready to run 2')
    if args.resume_from: # TODO Remove this
        full_model_path = os.path.join(cfg.OUTPUT_DIR, args.resume_from)
        net.load_state_dict(torch.load(full_model_path))
        print('Model loaded from {}'.format(full_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('DEVICE', device)

    try:
        if args.eval_training:
            model_checkpoints_eval_runner(net, cfg)
            final_model_evaluation_runner(net, cfg)
        else:
            final_model_evaluation_runner(net, cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


