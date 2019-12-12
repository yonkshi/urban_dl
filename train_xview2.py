import sys
from os import path
import os
from argparse import ArgumentParser
import datetime
import enum
import timeit

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as torch_data
from torchvision import transforms, utils
from tensorboardX import SummaryWriter
from coolname import generate_slug
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from tabulate import tabulate
import wandb

from debug_tools import __benchmark_init, benchmark
from unet import UNet
from unet.utils import SloveniaDataset, Xview2Dataset, Xview2Detectron2Dataset
from unet.augmentations import *

from experiment_manager.metrics import f1_score
from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config
from experiment_manager.loss import soft_dice_loss, soft_dice_loss_balanced
from eval_unet_xview2 import model_eval

# import hp

def train_net(net,
              cfg):

    log_path = cfg.OUTPUT_DIR
    writer = SummaryWriter(log_path)

    run_config = {}
    run_config['CONFIG_NAME'] = cfg.NAME
    run_config['device'] = device
    run_config['log_path'] = cfg.OUTPUT_DIR
    run_config['training_set'] = cfg.DATASETS.TRAIN
    run_config['test set'] = cfg.DATASETS.TEST
    run_config['epochs'] = cfg.TRAINER.EPOCHS
    run_config['learning rate'] = cfg.TRAINER.LR
    run_config['batch size'] = cfg.TRAINER.BATCH_SIZE
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))


    optimizer = optim.Adam(net.parameters(),
                          lr=cfg.TRAINER.LR,
                          weight_decay=0.0005)
    if cfg.MODEL.LOSS_TYPE == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
        balance_weight = [cfg.MODEL.NEGATIVE_WEIGHT, cfg.MODEL.POSITIVE_WEIGHT]
        balance_weight = torch.tensor(balance_weight).float().to(device)
        criterion = nn.CrossEntropyLoss(weight = balance_weight)
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceLoss':
        criterion = soft_dice_loss
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceBalancedLoss':
        criterion = soft_dice_loss_balanced

    net.to(device)

    __benchmark_init()
    global_step = 0
    epochs = cfg.TRAINER.EPOCHS
    start = timeit.default_timer()
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        net.train()
        trfm = []
        if cfg.AUGMENTATION.RESIZE: trfm.append( Resize(scale=cfg.AUGMENTATION.RESIZE_RATIO))
        if cfg.AUGMENTATION.CROP_TYPE == 'uniform': trfm.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
        elif cfg.AUGMENTATION.CROP_TYPE == 'importance': trfm.append(ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
        trfm.append(PIL2Torch())
        trfm = transforms.Compose(trfm)


        # reset the generators
        dataset = Xview2Detectron2Dataset(cfg.DATASETS.TRAIN[0],
                                          include_image_weight=True,
                                          image_oversampling = cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE,
                                          transform=trfm,
                                          legacy_mask_rasterization=cfg.DATALOADER.LEGACY_MASK_RASTERIZATION,
                                          )
        dataloader = torch_data.DataLoader(dataset,
                                           batch_size=cfg.TRAINER.BATCH_SIZE,
                                           num_workers=cfg.DATALOADER.NUM_WORKER,
                                           shuffle = cfg.DATALOADER.SHUFFLE,
                                           drop_last=True,
                                           )

        epoch_loss = 0
        benchmark('Dataset Setup')

        # mean AP, mean AUC, max F1
        mAP_set_train, mAUC_set_train, maxF1_train = [],[],[]
        loss_set, f1_set = [], []
        positive_pixels_set = [] # Used to evaluated image over sampling techniques
        for i, (x, y_gts, sample_name, image_weight) in enumerate(dataloader):
            # visualize_image(imgs, y_label, y_label, sample_name)
            # print('max_gpu_usage',torch.cuda.max_memory_allocated() / 10e9, ', max_GPU_cache_isage', torch.cuda.max_memory_cached()/10e9)
            optimizer.zero_grad()

            x = x.to(device)

            y_gts = y_gts.to(device)

            y_pred = net(x)

            if cfg.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
                # y_pred = y_pred # Cross entropy loss doesn't like single channel dimension
                y_gts = y_gts.long()# Cross entropy loss requires a long as target
            else:
                # For BCE loss, label needs to match the dimensions of network output
                y_gts = y_gts.unsqueeze(1)
            loss = criterion(y_pred, y_gts)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())
            positive_pixels_set.extend(image_weight.cpu().numpy())

            if global_step % 100 == 0 or global_step == 0:
                # time per 100 steps
                stop = timeit.default_timer()
                time_per_n_batches= stop - start

                if global_step % 10000 == 0 and global_step > 0:
                    check_point_name = f'cp_{global_step}.pkl'
                    save_path = os.path.join(log_path, check_point_name)
                    torch.save(net.state_dict(), save_path)

                # Averaged loss and f1 writer

                # writer.add_scalar('f1/train', np.mean(f1_set), global_step)

                max_mem, max_cache = gpu_stats()
                print(f'step {global_step},  avg loss: {np.mean(loss_set):.4f}, cuda mem: {max_mem} MB, cuda cache: {max_cache} MB, time: {time_per_n_batches:.2f}s',
                      flush=True)

                wandb.log({
                    'loss': np.mean(loss_set),
                    'gpu_memory': max_mem,
                    'time': time_per_n_batches,
                    'total_positive_pixels': np.mean(positive_pixels_set),
                    'step': global_step,
                })

                loss_set = []
                positive_pixels_set = []
                # f1_set = []
                if global_step % 1000 == 0:
                    figure, plt = visualize_image(x, y_pred, y_gts, sample_name)
                    wandb.log({"output_image": plt})
                    writer.add_figure('output_image/train', figure, global_step)

                optimizer.zero_grad()

                start = stop

            # torch.cuda.empty_cache()
            __benchmark_init()
            global_step += 1

        # Evaluation after each epoch
        model_eval(net, cfg, device, max_samples=100, step=global_step, epoch=epoch)
        model_eval(net, cfg, device, max_samples=100, run_type='TRAIN', step=global_step, epoch=epoch)





class LULC(enum.Enum):
    BACKGROUND = (0, 'Background', 'black')
    NO_DATA = (1, 'No Data', 'white')
    NO_DAMAGE = (2, 'No damage', 'xkcd:lime')
    MINOR_DAMAGE = (3, 'Minor Damage', 'yellow')
    MAJOR_DAMAGE = (4, 'Major Damage', 'orange')
    DESTROYED = (5, 'Destroyed', 'red')

    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3

lulc_cmap = ListedColormap([entry.color for entry in LULC])
lulc_norm = BoundaryNorm(np.arange(-0.5, 6, 1), lulc_cmap.N)

def gpu_stats():
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e6 # bytes to MB
    max_memory_cached = torch.cuda.max_memory_cached() /1e6


    return int(max_memory_allocated), int(max_memory_cached)


def visualize_image(input_image, output_segmentation, gt_segmentation, sample_name):

    # TODO This is slow, consider making this working in a background thread. Or making the entire tensorboardx work in a background thread
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(wspace=0, hspace=0)

    # input image
    img = toNp(input_image)
    # img = img[...,[2,1,0]] * 4.5 # BGR -> RGB and brighten
    img = np.clip(img, 0, 1)
    ax0 = fig.add_subplot(gs[0,0])
    ax0.set_title(sample_name[0])
    ax0.imshow(img)
    ax0.axis('off')

    # output segments
    out_seg = toNp(output_segmentation)
    out_seg_argmax = np.argmax(out_seg, axis=-1)
    ax1 = fig.add_subplot(gs[1, 0])
    print(out_seg_argmax.shape)
    ax1.imshow(out_seg_argmax.squeeze(), cmap = lulc_cmap, norm=lulc_norm,)
    ax1.set_title('output')
    ax1.axis('off')

    # ground truth
    gt = toNp_vanilla(gt_segmentation)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(gt.squeeze(), cmap=lulc_cmap, norm=lulc_norm,)
    ax2.set_title('ground_truth')
    ax2.axis('off')

    fig.tight_layout()
    plt.tight_layout()
    return fig, plt

def toNp_vanilla(t:torch.Tensor):
    return t[0,...].cpu().detach().numpy()

def toNp(t:torch.Tensor):
    # Pick the first item in batch
    return to_H_W_C(t)[0,...].cpu().detach().numpy()

def to_C_H_W(t:torch.Tensor):
    # From [B, H, W, C] to [B, C, H, W]
    if len(t.shape) <= 3: return t # if [B, H. W] then do nothing
    assert t.shape[1] == t.shape[2] and t.shape[3] != t.shape[2], 'are you sure this tensor is in [B, H, W, C] format?'
    return t.permute(0,3,1,2)

def to_H_W_C(t:torch.Tensor):
    # From [B, C, H, W] to [B, H, W, C]
    if len(t.shape) <= 3: return t # if [B, H. W] then do nothing
    assert t.shape[1] != t.shape[2], 'are you sure this tensor is in [B, C, H, W] format?'
    return t.permute(0,2,3,1)

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

if __name__ == '__main__':
    args = default_argument_parser().parse_known_args()[0]
    cfg = setup(args)

    out_channels = cfg.MODEL.OUT_CHANNELS
    net = UNet(cfg)

    if args.resume and args.resume_from:
        full_model_path = path.join(cfg.OUTPUT_DIR, args.model_path)
        net.load_state_dict(torch.load(full_model_path))
        print('Model loaded from {}'.format(full_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark = True # faster convolutions, but more memory

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        project='urban_dl',
        tags=['run'],
    )

    try:
        train_net(net, cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


