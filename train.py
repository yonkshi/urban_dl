import sys
import os
from optparse import OptionParser
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils import data as torch_data
from tensorboardX import SummaryWriter
from coolname import generate_slug

from eval import eval_net
from unet import UNet
from unet.utils import SloveniaDataset

DATASET_DIR = 'data/slovenia/slovenia2017.hdf5'
BATCH_SIZE = 1

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              device=torch.device('cpu'),
              img_scale=0.5):

    run_name = generate_slug(2) + '-' + datetime.datetime.today().strftime('%b-%d')
    log_path = 'logs/%s' % run_name
    writer = SummaryWriter(log_path)

    # TODO Save Run Config in Pandas
    # TODO Save

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        dataset = SloveniaDataset(DATASET_DIR)
        dataloader = torch_data.DataLoader(dataset,
                                           batch_size=BATCH_SIZE,
                                           pin_memory=True,
                                           num_workers=1,
                                           drop_last=True,
                                           )

        epoch_loss = 0

        for i, (imgs, true_masks) in enumerate(dataloader):

            global_step = epoch * batch_size + i

            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            masks_pred = net(imgs)
            loss = criterion(masks_pred, true_masks)
            epoch_loss += loss.item()

            print('loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Write things in
            writer.add_scalar('loss', loss, global_step)



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=6, n_classes=10)

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
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
