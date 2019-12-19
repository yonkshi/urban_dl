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
from os import makedirs

from torch.utils import data as torch_data
from torchvision import transforms, utils
from tabulate import tabulate
from debug_tools import __benchmark_init, benchmark

from unet import UNet
from unet.dataloader import Xview2Detectron2Dataset, Xview2Detectron2DamageLevelDataset
from experiment_manager.metrics import roc_score, f1_score, MultiThresholdMetric, MultiClassF1
from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config
from experiment_manager.utils import to_numpy
from experiment_manager.dataset import SimpleInferenceDataset
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from unet.augmentations import *
# import hp

def final_model_evaluation_runner(net, cfg):
    '''
    Runner that only concerns with only a single model,
    :return:
    '''
    import matplotlib.pyplot as plt
    print('=== Evaluating final model ===')
    # Setup

    F1_THRESH = torch.linspace(0, 1, 100).to(device)
    y_true_set = []
    y_pred_set = []
    measurer = MultiThresholdMetric(F1_THRESH)

    def evaluate(y_true, y_pred, img_filename):
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        y_true_set.append(y_true.cpu())
        y_pred_set.append(y_pred.cpu())

        measurer.add_sample(y_true, y_pred)

    inference_loop(net, cfg, device, evaluate)

    # ===
    # Collect for summary

    y_true_set = torch.cat(y_true_set, dim = 0).round()
    y_pred_set = torch.cat(y_pred_set, dim=0)

    y_true_set, y_pred_set = downsample_dataset_for_eval(y_true_set, y_pred_set)

    y_true_np = to_numpy(y_true_set.flatten())
    y_pred_np = to_numpy(y_pred_set.flatten())

    # F1 score
    print('Computing F1 vs thresholds', flush=True)
    f1 = measurer.compute_f1().cpu().numpy()
    plt.plot(np.arange(0, 100, 1, dtype=np.int32), f1)
    plt.ylabel('f1 score')
    plt.xlabel('threshold ')
    plt.title('F1 vs threshold curve')

    wandb.log({'f1 vs threshold': plt})
    wandb.log({
        'total False Negative': measurer.FN.cpu(),
        'total False Positive': measurer.FP.cpu(),
               })

    print('computing ROC curve', flush=True)
    # ROC curve
    fpr, tpr, thresh = roc_curve(y_true_np, y_pred_np)

    # Down sample roc curve or matplotlib won't like it
    num_ele = len(fpr)
    downsample_idx = np.linspace(0, num_ele, 1000, dtype=np.int32, endpoint=False)
    fpr_downsampled = fpr[downsample_idx]
    tpr_downsampled = tpr[downsample_idx]



    plt.plot(fpr_downsampled, tpr_downsampled)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.ylabel('true_positive rate')
    plt.xlabel('false_positive rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC curve')

    wandb.log({'roc': plt})


    print('done')

def model_checkpoints_eval_runner(net, cfg):

    checkpoint_files = list_and_sort_checkpoint_files()

    for cp_num, cp_file in checkpoint_files:
        print(' ==== checkpoint', cp_file)
        full_model_path = os.path.join(cfg.OUTPUT_DIR, cp_file)
        net.load_state_dict(torch.load(full_model_path))

        # TRAINING EVALUATION
        model_eval(net, cfg, device, run_type='TRAIN', step=cp_num)

        # TEST SET EVALUATION
        model_eval(net, cfg, device, run_type='TEST', step=cp_num)

def model_inference(net, cfg):
    '''
    This method is for running inference on the actual dataset
    :return:
    '''
    inference_dataset = cfg.DATASETS.INFERENCE[0]
    THRESHOLD = cfg.THRESH
    dataset = SimpleInferenceDataset(inference_dataset, downsample_scale= cfg.AUGMENTATION.RESIZE_RATIO)
    from PIL import Image

    def save_to_png(y_true, y_pred, img_filenames):
        # Y_pred activation

        # interp image if scaling was originally enabled
        if cfg.AUGMENTATION.RESIZE:
            upscale_ratio = 1 / cfg.AUGMENTATION.RESIZE_RATIO
            y_pred = torch.nn.functional.interpolate(y_pred,
                                                     scale_factor=upscale_ratio,
                                                     mode='bilinear')

        y_pred = (y_pred > THRESHOLD).type(torch.uint8)

        y_pred = y_pred.squeeze().cpu().numpy()
        img_filename = img_filenames[0]
        inference_dir = os.path.join(cfg.OUTPUT_DIR, 'predictions')
        os.makedirs(inference_dir, exist_ok=True)



        if img_filename.startswith('test'): # for the real dataset
            test, pre, num_png = str.split(img_filename, '_')
            num, png = str.split(num_png, '.')
            test_localization_num_pred = '_'.join([test,'localization', num, 'prediction'])
            img_filename = test_localization_num_pred + '.png'

            test_damage_num_pred = '_'.join([test, 'damage', num, 'prediction'])
            dmg_img_filename = test_damage_num_pred + '.png'

        img_save_dir = os.path.join(inference_dir, img_filename)
        dmg_img_save_dir = os.path.join(inference_dir, dmg_img_filename)
        im = Image.fromarray(y_pred, mode='L')
        im.save(img_save_dir)
        im.save(dmg_img_save_dir)



    inference_loop(net, cfg, device, save_to_png, dataset=dataset)

    return

def model_eval(net, cfg, device, run_type='TEST', max_samples = 1000, step=0, epoch=0, multi_class=False):
    '''
    Runner that is concerned with training changes
    :param run_type: 'train' or 'eval'
    :return:
    '''

    F1_THRESH = torch.linspace(0, 1, 100).to(device)
    y_true_set = []
    y_pred_set = []

    if multi_class:
        measurer = MultiClassF1(F1_THRESH)
    else:
        measurer = MultiThresholdMetric(F1_THRESH)
    def evaluate(y_true, y_pred, img_filename):
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


    # Due to interpolation
    y_true_set = torch.cat(y_true_set, dim = 0).round()
    y_pred_set = torch.cat(y_pred_set, dim=0)

    y_true_set, y_pred_set = downsample_dataset_for_eval(y_true_set, y_pred_set)

    y_true_np = to_numpy(y_true_set.flatten())
    y_pred_np = to_numpy(y_pred_set.flatten())

    # Average Precision
    print('Computing AP score ... ', end='', flush=True)
    ap = average_precision_score(y_true_np, y_pred_np)
    print(ap)

    set_name = 'test_set' if run_type == 'TEST' else 'training_set'
    wandb.log({f'{set_name} max F1': maxF1,
               f'{set_name} argmax F1': argmaxF1,
               f'{set_name} Average Precision': ap,
               f'{set_name} false positive rate': best_fpr,
               f'{set_name} false negative rate': best_fnr,
               'step': step,
               'epoch': epoch,
               })

def dmg_model_eval(net, cfg, device, run_type='TEST', max_samples = 1000, step=0, epoch=0, multi_class=False):
    '''
    Runner that is concerned with training changes
    :param run_type: 'train' or 'eval'
    :return:
    '''
    measurer = MultiClassF1()
    def evaluate(y_true, y_pred, img_filename):
        measurer.add_sample(y_true, y_pred)

    dset_source = cfg.DATASETS.TEST[0] if run_type == 'TEST' else cfg.DATASETS.TRAIN[0]

    trfm = []
    trfm.append(BGR2RGB())
    trfm.append(IncludeLocalizationMask())
    if cfg.AUGMENTATION.RESIZE: trfm.append(Resize(scale=cfg.AUGMENTATION.RESIZE_RATIO))
    trfm.append(Npy2Torch())
    if cfg.AUGMENTATION.ENABLE_VARI: trfm.append(VARI())
    trfm = transforms.Compose(trfm)

    dataset = Xview2Detectron2DamageLevelDataset(dset_source,
                          pre_or_post=cfg.DATASETS.PRE_OR_POST,
                          transform=trfm,)

    inference_loop(net, cfg, device, evaluate, batch_size=cfg.TRAINER.INFERENCE_BATCH_SIZE, run_type='TRAIN',  max_samples = max_samples, dataset = dataset)


    # Summary gathering ===

    print('Computing F1 score ... ', end=' ', flush=True)
    # Max of the mean F1 score

    # measurer = MultiThresholdMetric(y_true_set, y_pred_set, F1_THRESH)
    # Max F1

    total_f1, f1_per_class = measurer.compute_f1(include_bg=False)
    all_fpr, all_fnr = measurer.compute_basic_metrics()
    print(total_f1, flush=True)

    set_name = 'test_set' if run_type == 'TEST' else 'training_set'
    log_data = {f'{set_name} total F1': total_f1,
               'step': step,
               'epoch': epoch,
               }

    damage_levels = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    for f1, dmg in zip(f1_per_class, damage_levels):
        log_data[f'{set_name} {dmg} f1'] = f1

    damage_levels += ['negative class']
    for fpr, fnr, dmg in zip(all_fpr, all_fnr, damage_levels):
        log_data[f'{set_name} {dmg} false negative rate'] = fnr
        log_data[f'{set_name} {dmg} false positive rate'] = fpr

    wandb.log(log_data)

def downsample_dataset_for_eval(y_true, y_pred):
    # Full dataset is too big to compute for the CPU, so we down sample it
    num_samples = y_pred.shape[0]
    downsample_size = 100
    if num_samples > downsample_size: # only downsample if data size is huge
        down_idx = np.linspace(0, num_samples, downsample_size, endpoint=False, dtype=np.int)
        y_pred = y_pred[down_idx]
        y_true = y_true[down_idx]

    return y_true, y_pred

def inference_loop(net, cfg, device,
                    callback = None,
                    batch_size = 1,
                    run_type = 'TEST',
                    max_samples = 999999999,
                    dataset = None,
                    callback_include_x = False,

              ):

    net.to(device)
    net.eval()

    # reset the generators

    dset_source = cfg.DATASETS.TEST[0] if run_type == 'TEST' else cfg.DATASETS.TRAIN[0]
    if dataset is None:
        trfm = []
        if cfg.AUGMENTATION.RESIZE: trfm.append( Resize(scale=cfg.AUGMENTATION.RESIZE_RATIO))
        trfm.append(BGR2RGB())
        trfm.append(Npy2Torch())
        if cfg.AUGMENTATION.ENABLE_VARI: trfm.append(VARI())
        trfm = transforms.Compose(trfm)

        dataset = Xview2Detectron2Dataset(dset_source, pre_or_post=cfg.DATASETS.PRE_OR_POST, transform=trfm)

    dataloader = torch_data.DataLoader(dataset,
                                       batch_size=batch_size,
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

            if step % 100 == 0 or step == dataset_length-1:
                print(f'Processed {step+1}/{dataset_length}')

            if y_pred.shape[1] > 1: # multi-class
                # In Two class Cross entropy mode, positive classes are in Channel #2
                y_pred = torch.softmax(y_pred, dim=1)
            else:
                y_pred = torch.sigmoid(y_pred)

            if callback:
                if callback_include_x:
                    callback(imgs, y_label, y_pred, sample_name)
                else:
                    callback(y_label, y_pred, sample_name)


            if (max_samples is not None) and step >= max_samples:
                break

def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
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

def custom_argparse(parser):
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser.add_argument('-T',"--eval-type",
                        dest='eval_type',
                        default="final",
                        choices=['final', 'checkpoints', 'inference'],
                        help="select an evaluation type")
    parser.add_argument("--threshold",
                        dest='threshold',
                        default=0.01,
                        type=float,
                        help="select an evaluation type")
    return parser

if __name__ == '__main__':

    parser = default_argument_parser()
    args = custom_argparse(parser).parse_known_args()[0]
    cfg = setup(args)
    print('ready to run 0')
    print('ready to run 1')
    # torch.set_default_dtype(torch.float16)f
    out_channels = 1 if cfg.MODEL.BINARY_CLASSIFICATION else cfg.MODEL.OUT_CHANNELS
    net = UNet(cfg)
    print('ready to run 2')
    if args.resume_from: # TODO Remove this
        full_model_path = os.path.join(cfg.OUTPUT_DIR, args.resume_from)
        net.load_state_dict(torch.load(full_model_path))
        print('Model loaded from {}'.format(full_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('DEVICE', device)

    try:
        if args.eval_type == 'checkpoints':
            wandb.init(
                name=cfg.NAME,
                project='urban_dl',
                tags=['checkpoints_eval', 'final_model_eval'],
            )
            model_checkpoints_eval_runner(net, cfg)
            final_model_evaluation_runner(net, cfg)
        elif args.eval_type == 'final':
            wandb.init(
                name=f'{cfg.NAME} ({args.resume_from})',
                project='urban_dl',
                tags=['final_model_eval'],
            )
            final_model_evaluation_runner(net, cfg)
        elif args.eval_type == 'inference':
            model_inference(net, cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


