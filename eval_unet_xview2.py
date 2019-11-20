import sys
import os
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
# import hp

def train_net(net,
              num_dataloaders = 1,
              device=torch.device('cpu'),
              args = None,
              data_dir = 'data/xview2/xview2_sample.hdf5',
              log_dir = 'logs/',
              ):

    run_name = f'Inference on {args.model_path}'


    # TODO Save Run Config in Pandas

    run_config = {}
    run_config['run_name'] = run_name
    run_config['device'] = device
    run_config['log_path'] = log_dir
    run_config['data_dir'] = data_dir
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    net.to(device)
    F1_THRESH = torch.linspace(0, 1, 100)[None, :, None, None].to(device)  # 1d vec to [ B, Thresh, H, W ]
    __benchmark_init()
    net.eval()

    # reset the generators
    dataset = Xview2Detectron2Dataset(data_dir, 0)
    dataloader = torch_data.DataLoader(dataset,
                                       batch_size=1,
                                       num_workers=num_dataloaders,
                                       shuffle = True,
                                       drop_last=True,
                                       )

    epoch_loss = 0
    benchmark('Dataset Setup')
    f1s = []
    dataset_length = len(dataset)
    with torch.no_grad():
        for i, (imgs, y_label, sample_name) in enumerate(dataloader):
            # visualize_image(imgs, y_label, y_label, sample_name)
            # print('max_gpu_usage',torch.cuda.max_memory_allocated() / 10e9, ', max_GPU_cache_isage', torch.cuda.max_memory_cached()/10e9)
            imgs = imgs.to(device)
            y_label = y_label.to(device)
            print('prepping inference')
            y_pred = net(imgs)

            y_softmaxed = nn.Softmax2d()(y_pred)[:,1] # 1 = only positive labels
            print('Done inference')
            # Todo compute multiple F1 scores

            y_softmaxed = y_softmaxed[:,None,...] # [B, Thresh, H, W]
            y_softmaxed = torch.clamp(y_softmaxed - F1_THRESH + 0.5, 0, 1.0)
            f1 = f1_score(y_softmaxed, y_label, multi_threashold_mode=True)
            print('F1 Score')
            if i % 100 == 0 or i == dataset_length-1:
                print(f'Processed {i+1}/{dataset_length}')

            f1s.append(f1)
    # mean f1
    mF1 = torch.cat(f1s, 0)
    mF1 = torch.mean(mF1, dim=0)

    # Writing to TB
    write_path = os.path.join(args.log_dir, 'inference', args.model_path)
    writer = SummaryWriter(write_path)

    for i, val in enumerate(mF1):
        writer.add_scalar('Threshold vs F1', val, global_step=i)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-w', '--num-worker', dest='num_dataloaders', default=1,
                      type=int, help='number of dataloader workers')
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_argument('-m', '--model-path', dest='model_path',
                      default=False, help='load file model')
    parser.add_argument('-s', '--scale', dest='scale', type=float,
                      default=0.5, help='downscaling factor of the images')

    parser.add_argument('-d', '--data-dir', dest='data_dir', type=str,
                      default='data/xview2/xview2_sample.hdf5', help='dataset directory')
    parser.add_argument('-o', '--log-dir', dest='log_dir', type=str,
                      default='logs', help='logging directory')

    parser.add_argument( '--eval-only', dest='log_dir', type=str,
                      default='logs', help='logging directory')
    (options, args) = parser.parse_known_args()
    return options


if __name__ == '__main__':
    args = get_args()
    # torch.set_default_dtype(torch.float16)
    net = UNet(n_channels=3, n_classes=2)

    if args.model_path:
        full_model_path = os.path.join(args.log_dir, args.model_path)
        net.load_state_dict(torch.load(full_model_path))
        print('Model loaded from {}'.format(full_model_path))

    if args.gpu:
        device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
    else:
        device = torch.device('cpu')
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  device=device,
                  args = args,
                  num_dataloaders = args.num_dataloaders,
                  data_dir = args.data_dir,
                  log_dir = args.log_dir,
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


