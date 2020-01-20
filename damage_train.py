import sys
from os import path
import timeit
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data as torch_data
from torchvision import transforms, utils
import segmentation_models_pytorch as smp
from tabulate import tabulate
from sklearn.metrics import confusion_matrix as confmatrix
import wandb
import matplotlib.pyplot as plt

from unet import UNet
from unet.dataloader import Xview2Detectron2DamageLevelDataset
from unet.augmentations import *

from experiment_manager.args import default_argument_parser
from experiment_manager.metrics import MultiClassF1
from experiment_manager.config import new_config
from experiment_manager.loss import *
from eval_unet_xview2 import inference_loop

# import hp


def train_net(net,
              cfg):

    log_path = cfg.OUTPUT_DIR
    summarize_config(cfg)

    optimizer = optim.Adam(net.parameters(),
                           lr=cfg.TRAINER.LR,
                           weight_decay=0.0005)
    weighted_criterion = False
    if cfg.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
        criterion = cross_entropy_loss
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceMulticlassLoss':
        criterion = soft_dice_loss_multi_class
    elif cfg.MODEL.LOSS_TYPE == 'GeneralizedDiceLoss':
        criterion = generalized_soft_dice_loss_multi_class
    elif cfg.MODEL.LOSS_TYPE == 'JaccardLikeLoss':
        criterion = jaccard_like_loss_multi_class
    elif cfg.MODEL.LOSS_TYPE == 'ComboLoss':
        criterion = combo_loss
        weighted_criterion = cfg.TRAINER.CE_CLASS_BALANCE.ENABLED
        weights = 1 / torch.tensor(cfg.TRAINER.CE_CLASS_BALANCE.WEIGHTS)
        weights = weights.cuda()


    if cfg.MODEL.PRETRAINED.ENABLED:
        net = load_pretrained(net, cfg)

    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    bg_class = cfg.MODEL.BACKGROUND.TYPE
    trfm = build_transforms(cfg, for_training=True, use_gts_mask=cfg.DATASETS.LOCALIZATION_MASK.TRAIN_USE_GTS_MASK)
    dataset = Xview2Detectron2DamageLevelDataset(cfg.DATASETS.TRAIN[0],
                                                 pre_or_post='post',
                                                 include_image_weight=True,
                                                 background_class=bg_class,
                                                 transform=trfm)

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
        for i, batch in enumerate(dataloader):
            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            image_weight = batch['image_weight']

            optimizer.zero_grad()

            y_pred = net(x)
            if weighted_criterion:
                loss = criterion(y_pred, y_gts, weights)
            else:
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

            if global_step % 100 == 0 and global_step > 0:
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

        # Evaluation for multiclass F1 score
        dmg_model_eval(net, cfg, device, max_samples=100, step=global_step, epoch=epoch)
        dmg_model_eval(net, cfg, device, max_samples=100, run_type='TRAIN', step=global_step, epoch=epoch)

def dmg_model_eval(net, cfg, device, run_type='TEST', max_samples = 1000, step=0, epoch=0, use_confusion_matrix=False):
    '''
    Runner that is concerned with training changes
    :param run_type: 'train' or 'eval'
    :return:
    '''
    measurer = MultiClassF1(ignore_last_class=cfg.MODEL.BACKGROUND.TYPE=='new-class')

    confusion_matrix_with_bg = []
    confusion_matrix = []
    def evaluate(x, y_true, y_pred, img_filename):
        if not cfg.MODEL.BACKGROUND.MASK_OUTPUT:
            # No background class, manually mask out background
            localization_mask = x[:,[3]] # 3 is a hard coded mask index
            y_pred = localization_mask * y_pred
        measurer.add_sample(y_true, y_pred)
        if use_confusion_matrix:
            y_true_flat = y_true.argmax(dim=1).cpu().detach().flatten().numpy()
            y_pred_flat = y_pred.argmax(dim=1).cpu().detach().flatten().numpy()
            labels = [0, 1, 2, 3, 4] # 5 classes
            _mat = confmatrix(y_true_flat, y_pred_flat, labels = labels)
            confusion_matrix_with_bg.append(_mat)
    use_gts_mask = run_type == 'TRAIN' and cfg.DATASETS.LOCALIZATION_MASK.TRAIN_USE_GTS_MASK
    dset_source = cfg.DATASETS.TEST[0] if run_type == 'TEST' else cfg.DATASETS.TRAIN[0]

    trfm = build_transforms(cfg, use_gts_mask = use_gts_mask)
    bg_class = 'new-class' if cfg.MODEL.BACKGROUND.TYPE == 'new-class' else None
    dataset = Xview2Detectron2DamageLevelDataset(dset_source,
                                                 pre_or_post='post',
                                                 transform=trfm,
                                                 background_class=bg_class,)
    inference_loop(net, cfg, device, evaluate,
                   batch_size=cfg.TRAINER.INFERENCE_BATCH_SIZE,
                   run_type='TRAIN',
                   max_samples = max_samples,
                   dataset = dataset,
                   callback_include_x=True)

    # Summary gathering ===

    print('Computing F1 score ... ', end=' ', flush=True)
    # Max of the mean F1 score

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


    # Plot confusion matrix
    if use_confusion_matrix:
        normalized_cm = np.sum(confusion_matrix_with_bg, axis=0, keepdims=True)
        print('confusion_matrix', normalized_cm)
        # normalized_cm = confusion_matrix_with_bg / confusion_matrix_with_bg.sum(axis=0, keepdims=True)
        labels = ['no-damage', 'minor-damage', 'major-damage', 'destroyed', 'background',]

        fig, ax = plt.subplots()

        ax.matshow(normalized_cm, cmap='Blues')

        ax.set_yticks(np.arange(5))

        ax.set_yticklabels(labels)
        ax.set_xticklabels([''] + labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        fig.tight_layout()
        plt.title('confusion matrix')
        plt.savefig(f'{cfg.OUTPUT_DIR}/dmg_{run_type}_confusion_matrix_raw.png')
        log_data['confusion_matrix'] = plt

    wandb.log(log_data)

def load_pretrained(net:nn.Module, cfg):
    p_cfg = cfg.MODEL.PRETRAINED
    dirs = os.path.abspath(cfg.OUTPUT_DIR).split('/')
    dirs = dirs[:-2]
    dirs += ['unet', p_cfg.NAME, p_cfg.CP_FILE] # remove ./dmg/run_name/ -> ./
    cp_path = '/'.join(dirs)

    loaded_dict = torch.load(cp_path)

    state = OrderedDict()
    for key, tensor in loaded_dict.items():
        # Skip output layer
        if key.startswith('outc'): continue
        # Skip input layer
        if not p_cfg.INCLUDE_INPUT_LAYER and key.startswith('inc'): continue
        # Skip decoder
        if p_cfg.ENCODER_ONLY and key.startswith('up_seq'): continue

        state[key] = tensor

    # Because merge state dict
    full_state = net.state_dict()
    full_state.update(state)
    net.load_state_dict(full_state)

    print('Pretrained model loaded! ', cp_path, flush=True)
    return net

def build_transforms(cfg, for_training=False, use_gts_mask = False):
    trfm = []
    trfm.append(BGR2RGB())

    if cfg.DATASETS.LOCALIZATION_MASK.ENABLED: trfm.append(IncludeLocalizationMask(use_gts_mask))
    if cfg.DATASETS.INCLUDE_PRE_DISASTER: trfm.append(StackPreDisasterImage())
    if cfg.AUGMENTATION.RESIZE: trfm.append(Resize(scale=cfg.AUGMENTATION.RESIZE_RATIO))
    if cfg.AUGMENTATION.CROP_TYPE == 'uniform' and for_training:
        trfm.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    elif cfg.AUGMENTATION.CROP_TYPE == 'importance':
        trfm.append(ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    trfm.append(Npy2Torch())
    if cfg.AUGMENTATION.ENABLE_VARI: trfm.append(VARI())
    trfm = transforms.Compose(trfm)
    return trfm

def combo_loss(p, y, class_weights=None):
    y_ = y.argmax(dim=1).long()
    loss = F.cross_entropy(p, y_, weight=class_weights) + soft_dice_loss_multi_class(p, y)
    return loss


def image_sampling_weight(dataset_metadata):
    print('performing oversampling...', end='', flush=True)
    EMPTY_IMAGE_BASELINE = 1000
    image_p = np.array([image_desc['post']['image_weight'] for image_desc in dataset_metadata]) + EMPTY_IMAGE_BASELINE
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

def cross_entropy_loss(pred, y):
    y = y.argmax(dim=1).long()
    return F.cross_entropy(pred, y)

def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/damage_detection/{args.config_file}.yaml')
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
    if cfg.MODEL.BACKBONE.ENABLED:
        if cfg.MODEL.COMPLEX_ARCHITECTURE.ENABLED:
            if cfg.MODEL.COMPLEX_ARCHITECTURE.TYPE == 'pspnet':
                net = smp.PSPNet(cfg.MODEL.BACKBONE.TYPE,
                               encoder_weights=None,
                               encoder_depth=3,
                               in_channels=cfg.MODEL.IN_CHANNELS,
                               classes=cfg.MODEL.OUT_CHANNELS
                               )
        else:
            net = smp.Unet(cfg.MODEL.BACKBONE.TYPE,
                           encoder_weights=None,
                           encoder_depth=5,
                           decoder_channels = [512,256,128,64,32],
                           in_channels= cfg.MODEL.IN_CHANNELS,
                           classes=cfg.MODEL.OUT_CHANNELS
            )
    else:
        net = UNet(cfg)

    if args.resume and args.resume_from:
        full_model_path = path.join(cfg.OUTPUT_DIR, args.model_path)
        net.load_state_dict(torch.load(full_model_path))
        print('Model loaded from {}'.format(full_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark = True # faster convolutions, but more memory

    print('=== Runnning on device: p', device)


    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:

        if args.eval_only:
            wandb.init(
                name=cfg.NAME,
                project='urban_dl',
                tags=['eval', 'dmg'],
            )
            dmg_model_eval(net, cfg, device, run_type='TEST', max_samples=1000,  use_confusion_matrix=True)
            dmg_model_eval(net, cfg, device, run_type='TRAIN', max_samples=1000, use_confusion_matrix=True)
        else:
            wandb.init(
                name=cfg.NAME,
                project='urban_dl',
                tags=['run', 'dmg'],
                reinit=True
            )
            train_net(net, cfg)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


