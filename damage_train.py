import sys
from os import path
import timeit

import torch.nn as nn
from torch import optim
from torch.utils import data as torch_data
from torch.nn import functional as F
from torchvision import transforms, utils

from tabulate import tabulate
import wandb

from unet import UNet
from unet.utils import Xview2Detectron2Dataset
from unet.augmentations import *

from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config
from experiment_manager.loss import soft_dice_loss, soft_dice_loss_balanced, jaccard_like_loss, jaccard_like_balanced_loss, soft_dice_loss_multi_class
from eval_unet_xview2 import model_eval

# import hp

def train_net(net,
              cfg):

    log_path = cfg.OUTPUT_DIR
    summarize_config(cfg)

    optimizer = optim.Adam(net.parameters(),
                          lr=cfg.TRAINER.LR,
                          weight_decay=0.0005)
    if cfg.MODEL.LOSS_TYPE == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceMulticlassLoss':
        criterion = soft_dice_loss_multi_class
    net.to(device)

    trfm = []
    if cfg.AUGMENTATION.RESIZE: trfm.append(Resize(scale=cfg.AUGMENTATION.RESIZE_RATIO))
    if cfg.AUGMENTATION.CROP_TYPE == 'uniform':
        trfm.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    trfm.append(Npy2Torch())
    if cfg.AUGMENTATION.ENABLE_VARI: trfm.append(VARI())
    trfm = transforms.Compose(trfm)

    # reset the generators
    dataset = Xview2Detectron2Dataset(cfg.DATASETS.TRAIN[0],
                                      pre_or_post=cfg.DATASETS.PRE_OR_POST,
                                      include_image_weight=True,
                                      transform=trfm,)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
    }

    # sampler
    if cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'simple':
        image_p = image_sampling_weight(dataset.dataset_metadata)
        sampler = torch_data.WeightedRandomSampler(weights=image_p, num_samples=len(image_p))
        dataloader_kwargs['sampler'] = sampler
        dataloader_kwargs['shuffle'] = False

    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    max_epochs = cfg.TRAINER.EPOCHS
    global_step = 0
    for epoch in range(max_epochs):
        start = timeit.default_timer()
        print('Starting epoch {}/{}.'.format(epoch + 1, max_epochs))
        epoch_loss = 0

        net.train()
        loss_set, f1_set = [], []
        positive_pixels_set = [] # Used to evaluated image over sampling techniques
        for i, (x, y_gts, sample_name, image_weight) in enumerate(dataloader):
            optimizer.zero_grad()

            x = x.to(device)
            y_gts = y_gts.to(device)
            y_pred = net(x)

            # For BCE loss, label needs to match the dimensions of network output
            y_gts = y_gts.unsqueeze(1)
            loss = criterion(y_pred, y_gts)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())
            positive_pixels_set.extend(image_weight.cpu().numpy())

            if global_step % 10000 == 0 and global_step > 0:
                check_point_name = f'cp_{global_step}.pkl'
                save_path = os.path.join(log_path, check_point_name)
                torch.save(net.state_dict(), save_path)

            if global_step % 100 == 0 or global_step == 0:
                # time per 100 steps
                stop = timeit.default_timer()
                time_per_n_batches= stop - start

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
                start = stop

            # torch.cuda.empty_cache()
            global_step += 1

        if epoch % 2 == 0:
            # Evaluation after every other epoch
            model_eval(net, cfg, device, max_samples=100, step=global_step, epoch=epoch)
            model_eval(net, cfg, device, max_samples=100, run_type='TRAIN', step=global_step, epoch=epoch)



def image_sampling_weight(dataset_metadata):
    print('performing oversampling...', end='', flush=True)
    EMPTY_IMAGE_BASELINE = 1000
    image_p = np.array([image_desc['image_weight'] for image_desc in dataset_metadata]) + EMPTY_IMAGE_BASELINE
    print('done', flush=True)
    # normalize to [0., 1.]
    image_p = image_p
    return image_p

def gpu_stats():
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e6 # bytes to MB
    max_memory_cached = torch.cuda.max_memory_cached() /1e6
    return int(max_memory_allocated), int(max_memory_cached)

def summarize_config(cfg):
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


def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/damage_dectection/{args.config_file}.yaml')
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
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        train_net(net, cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


