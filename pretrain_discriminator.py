import sys
from os import path

import timeit

import torch.nn as nn
from torch import optim
from torch.utils import data as torch_data
from torch.nn import functional as F
from torchvision import transforms, utils
import segmentation_models_pytorch as smp
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from ignite.contrib.metrics import GpuInfo

from tabulate import tabulate
import wandb

from unet import UNet
from unet.dataloader import Xview2Detectron2DiscriminatorPretrainDataset
from unet.augmentations import *
from unet.descriminator_model import RefinementDescriminator, GradientReversal, RevGrad

from experiment_manager.args import default_argument_parser
from experiment_manager.config import new_config
from experiment_manager.loss import soft_dice_loss, soft_dice_loss_balanced, jaccard_like_loss, jaccard_like_balanced_loss
from eval_unet_xview2 import model_eval

# import hp

def train_net(descriminator_net,
              cfg,
              ):

    log_path = cfg.OUTPUT_DIR

    print_config(cfg)

    optimizer = optim.Adam(descriminator_net.parameters(),

                          lr=cfg.TRAINER.LR,
                          weight_decay=0.0005)
    criterion = build_loss
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), " GPUs!")
        descriminator_net = nn.DataParallel(descriminator_net)

    descriminator_net.to(device)


    trfm = []
    trfm = build_transforms(cfg, trfm)

    # reset the generators
    dataset = Xview2Detectron2DiscriminatorPretrainDataset(cfg.DATASETS.TRAIN[0],
                                                           pre_or_post='pre',
                                      transform=trfm,
                                      )

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }

    # sampler
    if cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'simple':
        image_p = image_sampling_weight(dataset.dataset_metadata)
        sampler = torch_data.WeightedRandomSampler(weights=image_p, num_samples=len(image_p))
        dataloader_kwargs['sampler'] = sampler
        dataloader_kwargs['shuffle'] = False

    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # Ignite code begins
    timer = Timer(average=False)
    def step(engine, batch):

        descriminator_net.train()
        optimizer.zero_grad()

        pred = batch['predicted'].to(device)
        gts = batch['gts'].to(device)
        gts_class = descriminator_net(gts)
        pred_class = descriminator_net(pred)
        loss = criterion(pred_class, gts_class)

        loss.backward()
        optimizer.step()

        return { 'loss': loss.item(), }


    trainer = Engine(step)

    # Metrics
    RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'avg_loss')
    GpuInfo().attach(trainer, 'gpu')

    # measurement timerq
    timer.attach(trainer,
                start=Events.EPOCH_STARTED,
                resume=Events.ITERATION_STARTED,
                pause=Events.ITERATION_COMPLETED)

    # Checkpoint handler
    cp_handler = ModelCheckpoint(log_path, 'cp', create_dir=True, n_saved=None, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=8), cp_handler, {'model': discriminator_net})

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def logging_function(engine:Engine):

        global_step = engine.state.iteration
        loss = engine.state.metrics['avg_loss']
        gpu_mem = engine.state.metrics['gpu:0 mem(%)']
        time = timer.value()

        print(
            f'step {global_step},  avg loss: {loss:.4f}, gpu mem: {gpu_mem}%, time: {time:.2f}s',
            flush=True)

        wandb.log({
            'loss': loss,
            'gpu_memory': gpu_mem,
            'time': time,
            'step': global_step,
        })

    trainer.run(dataloader, max_epochs=1000)


def print_config(cfg):
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


def build_loss(pred, gts):
    target = torch.zeros_like(pred.squeeze())
    loss1 = F.binary_cross_entropy_with_logits(pred.squeeze(), target)
    target = torch.ones_like(pred.squeeze())
    loss2 = F.binary_cross_entropy_with_logits(gts.squeeze(), target)
    loss = (loss1 + loss2)/2
    return loss


def build_transforms(cfg, trfm):
    if cfg.AUGMENTATION.CROP_TYPE == 'uniform':
        trfm.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    elif cfg.AUGMENTATION.CROP_TYPE == 'importance':
        trfm.append(ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    if cfg.AUGMENTATION.RANDOM_FLIP_ROTATE: trfm.append(RandomFlipRotate())
    trfm.append(Npy2Torch())
    trfm = transforms.Compose(trfm)
    return trfm


def image_sampling_weight(dataset_metadata):
    print('performing oversampling...', end='', flush=True)
    EMPTY_IMAGE_BASELINE = 1000
    image_p = np.array([image_desc['pre']['image_weight'] for image_desc in dataset_metadata]) + EMPTY_IMAGE_BASELINE
    print('done', flush=True)
    # normalize to [0., 1.]
    image_p = image_p
    return image_p


def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/discriminator/{args.config_file}.yaml')
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
    discriminator_net = RefinementDescriminator(cfg)

    if args.resume and args.resume_from:
        full_model_path = path.join(cfg.OUTPUT_DIR, args.model_path)
        discriminator_net.load_state_dict(torch.load(full_model_path))
        print('Model loaded from {}'.format(full_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark = True # faster convolutions, but more memory

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        project='urban_dl',
        tags=['run', 'discriminator'],
    )
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        train_net(discriminator_net, cfg)
    except KeyboardInterrupt:
        torch.save(discriminator_net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


