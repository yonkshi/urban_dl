import sys
import os
from os import path
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils import data as torch_data
from tabulate import tabulate
from debug_tools import __benchmark_init, benchmark
from unet import UNet
from unet.utils import Xview2Detectron2Dataset
from experiment_manager.metrics import f1_score
from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config
# import hp

def train_net(net, cfg
              ):


    run_config = {}
    run_config['run_name'] = cfg.NAME
    run_config['device'] = device
    run_config['log_path'] = cfg.OUTPUT_DIR
    run_config['data_dir'] = cfg.DATASETS.TRAIN
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    net.to(device)

    F1_THRESH = torch.linspace(0, 1, 100)[None, :, None, None].to(device)  # 1d vec to [ B, Thresh, H, W ]
    __benchmark_init()
    net.eval()

    # reset the generators
    print('loading dataset')
    dataset = Xview2Detectron2Dataset(cfg.DATASETS.TRAIN[0], 0)
    dataloader = torch_data.DataLoader(dataset,
                                       batch_size=1,
                                       num_workers=cfg.DATALOADER.NUM_WORKER,
                                       shuffle = cfg.DATALOADER.SHUFFLE,
                                       drop_last=True,
                                       )

    epoch_loss = 0
    benchmark('Dataset Setup')
    f1s = []
    dataset_length = len(dataset)
    print('dataset loaded')
    with torch.no_grad():
        for i, (imgs, y_label, sample_name) in enumerate(dataloader):
            # visualize_image(imgs, y_label, y_label, sample_name)
            # print('max_gpu_usage',torch.cuda.max_memory_allocated() / 10e9, ', max_GPU_cache_isage', torch.cuda.max_memory_cached()/10e9)
            imgs = imgs.to(device)
            y_label = y_label.to(device)
            y_pred = net(imgs)
            print(y_pred.shape)
            y_softmaxed = y_pred #nn.Softmax2d()(y_pred)[:,1] # 1 = only positive labels

            # Todo compute multiple F1 scores
            # y_softmaxed = y_softmaxed[:,None,...] # [B, Thresh, H, W]
            y_softmaxed = torch.clamp(y_softmaxed - F1_THRESH + 0.5, 0, 1.0)
            f1 = f1_score(y_softmaxed, y_label, multi_threashold_mode=True)
            if i % 100 == 0 or i == dataset_length-1:
                print(f'Processed {i+1}/{dataset_length}')

            f1s.append(f1)
    # mean f1
    mF1 = torch.cat(f1s, 0)
    mF1 = torch.mean(mF1, dim=0)

    # Writing to TB
    write_path = os.path.join(cfg.OUTPUT_DIR, 'inference', args.resume_from)
    writer = SummaryWriter(write_path)

    for i, val in enumerate(mF1):
        writer.add_scalar('Threshold vs F1', val, global_step=i)

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

    # torch.set_default_dtype(torch.float16)f
    out_channels = 1 if cfg.MODEL.BINARY_CLASSIFICATION else cfg.MODEL.OUT_CHANNELS
    net = UNet(n_channels=cfg.MODEL.IN_CHANNELS, n_classes=out_channels)

    if args.resume and args.resume_from: # TODO Remove this
        full_model_path = os.path.join(cfg.OUTPUT_DIR, args.resume_from)
        net.load_state_dict(torch.load(full_model_path))
        print('Model loaded from {}'.format(full_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_net(net,cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


