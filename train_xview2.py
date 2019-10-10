import sys
from os import path
import os
from argparse import ArgumentParser
import datetime
import enum

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as torch_data
from tensorboardX import SummaryWriter
from coolname import generate_slug
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from tabulate import tabulate

from debug_tools import __benchmark_init, benchmark
from unet import UNet
from unet.utils import SloveniaDataset, Xview2Dataset
from experiment_manager.metrics import f1_score
# import hp

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.05,
              save_cp=True,
              num_dataloaders = 1,
              device=torch.device('cpu'),
              data_dir = 'data/xview2/xview2_sample.hdf5',
              log_dir = 'logs/',
              img_scale=0.5):

    run_name = datetime.datetime.today().strftime('%b-%d') + '-' + generate_slug(2)
    #log_path = 'logs/%s' % run_name
    log_path = path.join(log_dir, run_name)
    writer = SummaryWriter(log_path)

    # TODO Save Run Config in Pandas

    run_config = {}
    run_config['run_name'] = run_name
    run_config['device'] = device
    run_config['log_path'] = log_path
    run_config['data_dir'] = data_dir
    run_config['epochs'] = epochs
    run_config['learning rate'] = lr
    run_config['batch size'] = batch_size
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))


    optimizer = optim.Adam(net.parameters(),
                          lr=lr,
                          weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss()

    net.to(device)

    __benchmark_init()
    global_step = 0
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        net.train()

        # reset the generators
        dataset = Xview2Dataset(data_dir, epoch)
        dataloader = torch_data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           pin_memory=True,
                                           num_workers=num_dataloaders,
                                           drop_last=True,
                                           )

        epoch_loss = 0
        benchmark('Dataset Setup')

        for i, (imgs, y_label, sample_name, input_val, label_val, sample_name_val) in enumerate(dataloader):

            optimizer.zero_grad()

            imgs = imgs.to(device)
            y_label = y_label.to(device)

            y_pred = net(imgs)

            loss = criterion(y_pred, y_label)
            epoch_loss += loss.item()

            print('loss', loss.item())
            loss.backward()
            optimizer.step()

            # Write things in
            if global_step % 10 == 0 or global_step < 5:
                if global_step % 100 == 0:
                    print(f'\n======== COMPLETED epoch{epoch}, global step{global_step} ')
                if global_step % 60 == 0:
                    writer.add_histogram('output_categories', y_pred.detach())
                # Save checkpoints
                if global_step % 5000 == 0 and global_step > 0:
                    check_point_name = f'{run_name}_{global_step}.pkl'
                    save_path = os.path.join(log_path, check_point_name)
                    torch.save(net.state_dict(), save_path)

                writer.add_scalar('loss/train', loss.item(), global_step)
                figure, plt = visualize_image(imgs, y_pred, y_label, sample_name)
                writer.add_figure('output_image/train', figure, global_step)

                y_pred_binary = torch.argmax(y_pred, dim=1)
                f1 = f1_score(y_pred_binary, y_label)
                if global_step % 1000 == 0 and global_step > 0:
                    figure_name = f'{run_name}_{global_step}.png'
                    save_path = os.path.join(log_path, figure_name)
                    plt.savefig(save_path)
                writer.add_scalar('f1/train', f1, global_step)

                optimizer.zero_grad()
                # Test set
                input_val = input_val.to(device)
                label_val = label_val.to(device)
                y_pred_validation = net(input_val)
                f1 = f1_score(y_pred_validation, label_val)
                writer.add_scalar('f1/test', f1, global_step)
                figure, plt = visualize_image(input_val, y_pred_validation, label_val, sample_name_val)
                writer.add_figure('output_image/test', figure, global_step)



            # torch.cuda.empty_cache()
            __benchmark_init()
            global_step += 1



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
    assert t.shape[1] == t.shape[2] and t.shape[3] != t.shape[2], 'are you sure this tensor is in [B, H, W, C] format?'
    return t.permute(0,3,1,2)

def to_H_W_C(t:torch.Tensor):
    # From [B, C, H, W] to [B, H, W, C]
    assert t.shape[1] != t.shape[2], 'are you sure this tensor is in [B, C, H, W] format?'
    return t.permute(0,2,3,1)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', dest='epochs', default=5, type=int,
                      help='number of epochs')
    parser.add_argument('-b', '--batch_size', dest='batchsize', default=1,
                      type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', dest='lr', default=0.001,
                      type=float, help='learning rate')
    parser.add_argument('-w', '--num-worker', dest='num_dataloaders', default=1,
                      type=int, help='number of dataloader workers')
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_argument('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_argument('-s', '--scale', dest='scale', type=float,
                      default=0.5, help='downscaling factor of the images')

    parser.add_argument('-d', '--data-dir', dest='data_dir', type=str,
                      default='data/xview2/xview2_sample.hdf5', help='dataset directory')
    parser.add_argument('-o', '--log-dir', dest='log_dir', type=str,
                      default='logs', help='logging directory')

    (options, args) = parser.parse_known_args()
    return options


if __name__ == '__main__':
    args = get_args()
    # torch.set_default_dtype(torch.float16)
    net = UNet(n_channels=3, n_classes=2)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
    else:
        device = torch.device('cpu')
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  num_dataloaders = args.num_dataloaders,
                  data_dir = args.data_dir,
                  log_dir = args.log_dir,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


