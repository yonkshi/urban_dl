import sys
from os import path
import timeit
import platform
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data as torch_data
from torchvision import transforms, utils
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import wandb
import optuna

import matplotlib.pyplot as plt

import eval_unet_xview2
from unet import UNet, Dpn92_Unet_Double, SeNet154_Unet_Double, SeResNext50_Unet_Double
from unet.dataloader import Xview2Detectron2DamageLevelDataset
from unet.augmentations import *
from unet import schedulers

from experiment_manager.args import default_argument_parser
from experiment_manager.metrics import MultiClassF1
from experiment_manager.config import new_config
from experiment_manager.loss import *
from eval_unet_xview2 import inference_loop
from eval_util.xview2_scoring import RowPairCalculator, XviewMetrics

# import hp


def train_net(net, cfg, device, trial: optuna.Trial=None):

    log_path = cfg.OUTPUT_DIR
    summarize_config(cfg, device)
    with open(path.join(cfg.OUTPUT_DIR, cfg.CONFIG + '.yaml'), 'w') as f:
        f.write(cfg.dump())

    optimizer = optim.AdamW(net.parameters(),
                            lr=cfg.TRAINER.LR,
                            weight_decay=cfg.TRAINER.WD,
                            betas=(cfg.TRAINER.B1, cfg.TRAINER.B2))
    scheduler = schedulers.scheduler_from_cfg(cfg, optimizer)

    weighted_criterion = cfg.TRAINER.CE_CLASS_BALANCE.ENABLED
    if weighted_criterion: # Also works with combo loss
        weights = 1 / torch.tensor(cfg.TRAINER.CE_CLASS_BALANCE.WEIGHTS)
        weights = weights.cuda()

    if cfg.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
        criterion = cross_entropy_loss
    elif cfg.MODEL.LOSS_TYPE == 'WeightedCrossEntropyLoss':
        criterion = weighted_ce_loss
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceMulticlassLoss':
        criterion = soft_dice_loss_multi_class
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceMulticlassLossDebug':
        criterion = soft_dice_loss_multi_class_debug
    elif cfg.MODEL.LOSS_TYPE == 'GeneralizedDiceLoss':
        criterion = generalized_soft_dice_loss_multi_class
    elif cfg.MODEL.LOSS_TYPE == 'JaccardLikeLoss':
        criterion = jaccard_like_loss_multi_class
    elif cfg.MODEL.LOSS_TYPE == 'ComboLoss':
        criterion = combo_loss
    elif cfg.MODEL.LOSS_TYPE == 'MeanSquareError':
        criterion = mean_square_error
    else:
        raise ValueError(f'Unknown cfg.MODEL.LOSS_TYPE: {cfg.MODEL.LOSS_TYPE}')


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
                                                 label_format=cfg.DATASETS.LABEL_FORMAT,
                                                 transform=trfm)

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
    elif cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'per_class':
        image_p = class_uniform_image_sampling_weight(dataset.dataset_metadata)
        sampler = torch_data.WeightedRandomSampler(weights=image_p, num_samples=len(image_p))
        dataloader_kwargs['sampler'] = sampler
        dataloader_kwargs['shuffle'] = False
    else:
        raise ValueError(f'Unknown cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE: {cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE}')

    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    class_labels = {
        0: "no damage",
        1: "minor damage",
        2: "major damage",
        3: "destroyed",
        4: "background",
    }

    max_epochs = cfg.TRAINER.EPOCHS
    global_step = 0 if not cfg.DEBUG else 10000
    log_step = 0
    val_moving_avg = 0.0
    start = timeit.default_timer()
    for epoch in range(max_epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, max_epochs))
        epoch_loss = 0

        net.train()
        loss_set, f1_set = [], []
        loss_component_set = []
        positive_pixels_set = [] # Used to evaluated image over sampling techniques
        for i, batch in enumerate(dataloader):
            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            image_weight = batch['image_weight']

            optimizer.zero_grad()

            y_pred = net(x)
            ce_loss = 0
            dice_loss = 0
            if weighted_criterion:
                loss, (ce_loss, dice_loss) = criterion(y_pred, y_gts, weights)
            else:
                loss = criterion(y_pred, y_gts)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())
            # loss_component_set.append(loss_component.cpu().detach().numpy())
            positive_pixels_set.extend(image_weight.cpu().numpy())

            if global_step - log_step >= 100:
                log_step = global_step
                stop = timeit.default_timer()
                time_per_n_batches= stop - start

                max_mem, max_cache = gpu_stats()

                print(f'step {global_step},  avg loss: {np.mean(loss_set):.4f}, cuda mem: {max_mem} MB, cuda cache: {max_cache} MB, time: {time_per_n_batches:.2f}s',
                      flush=True)

                log_data = {
                    'loss': np.mean(loss_set),
                    'ce_component_loss': ce_loss,
                    'dice_component_loss': dice_loss,
                    'gpu_memory': max_mem,
                    'time': time_per_n_batches,
                    'total_positive_pixels': np.mean(positive_pixels_set),
                    'step': global_step,
                    'lr': optimizer.param_groups[0]['lr'],
                }

                wandb.log(log_data, step=global_step)

                loss_set = []
                positive_pixels_set = []
                start = stop

                if cfg.DEBUG:
                    break

            # torch.cuda.empty_cache()
            global_step += cfg.TRAINER.BATCH_SIZE
            scheduler.step(global_step)

        if epoch % 10 == 0:
            check_point_name = f'cp_{global_step}.pkl'
            save_path = os.path.join(log_path, check_point_name)
            torch.save(net.state_dict(), save_path)

        if cfg.DATASETS.LABEL_FORMAT == 'ordinal':
            mask_data = eval_unet_xview2.ordinal_to_one_hot(y_pred)
            mask_data_gt = eval_unet_xview2.ordinal_to_one_hot(batch['y'])
        else:
            mask_data = y_pred
            mask_data_gt = batch['y']
        mask_data = mask_data[0].argmax(dim=0).detach().to('cpu').numpy()
        mask_data_gt = mask_data_gt[0].argmax(dim=0).numpy()

        # Log an image after every epoch
        wandb.log({'example_image': wandb.Image(batch['x'][0, 0:3].numpy().transpose(1, 2, 0), masks={
            "predictions": {
                "mask_data": mask_data,
                "class_labels": class_labels,
            },
            "groud_truth": {
                "mask_data": mask_data_gt,
                "class_labels": class_labels,
            }})},
                  step=global_step)

        # Evaluation for multiclass F1 score
        print('Validation set:')
        val_f1 = dmg_model_eval(net, cfg, device, global_step, max_samples=100, run_type='VALIDATION', step=global_step, epoch=epoch, use_confusion_matrix=False)
        val_moving_avg = val_f1 * 0.3 +  val_moving_avg * 0.7
        print('Training set:')
        dmg_model_eval(net, cfg, device, global_step, max_samples=100, run_type='TRAIN', step=global_step, epoch=epoch, use_confusion_matrix=False)


        # Check if we have to prune if in a trial
        if trial:
            trial.report(val_moving_avg, step=global_step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if cfg.DEBUG:
            return val_f1

    check_point_name = f'cp_{global_step}_final.pkl'
    save_path = os.path.join(log_path, check_point_name)
    torch.save(net.state_dict(), save_path)

    # Final evaluation
    dmg_model_eval(net, cfg, device, global_step, max_samples=None, run_type='TEST', step=global_step, epoch=epoch,
                   use_confusion_matrix=True, include_disaster_type_breakdown=True)
    return dmg_model_eval(net, cfg, device, global_step, max_samples=None, step=global_step, epoch=epoch, use_confusion_matrix=True, include_disaster_type_breakdown=True)

def dmg_model_eval(net, cfg, device, global_step,
                   run_type='VALIDATION',
                   max_samples = None,
                   step=0, epoch=0,
                   use_confusion_matrix=False,
                   include_component_f1=False,
                   include_disaster_type_breakdown = False):
    '''
    Runner that is concerned with training changes
    :param run_type: 'train' or 'eval'
    :return:
    '''
    measurer = MultiClassF1(ignore_last_class=cfg.MODEL.BACKGROUND.TYPE=='new-class')
    disaster_type_measurers = {}

    confusion_matrix_with_bg = []
    confusion_matrices_by_disaster_type = {}
    component_f1 = []
    allrows = []
    def evaluate(x, y_true, y_pred, img_filenames):

        # Only enable me if testing pretrained winner model where bg is class 0 !!
        # y_pred = y_pred[:,[1,2,3,4,0]]

        if cfg.MODEL.BACKGROUND.MASK_OUTPUT:
            # No background class, manually mask out background
            localization_mask = x[:,[3]] # 3 is a hard coded mask index
            y_pred = localization_mask * y_pred
        tp, tn, fp, fn = measurer.add_sample(y_true, y_pred)

        # === Component F1 score
        if include_component_f1:
            loss, loss_component = soft_dice_loss_multi_class_debug(y_pred, y_true)
            component_f1.append(loss_component.cpu().detach().numpy())

        # === Official F1 evaluation
        y_pred_np = y_pred.argmax(dim=1).cpu().detach().numpy() # One-hot to not-one-hot
        y_true_np = y_true.argmax(dim=1).cpu().detach().numpy()
        # loc_row is not being used, but to keep original code intact, we keep it here

        row = RowPairCalculator.get_row_pair(np.zeros([2]), y_pred_np, np.zeros([2]), y_true_np)
        allrows.append(row)

        # # #  DEBUGGED TP FP FN
        # for i in range(4):
        #     cls = i * 3
        #     print(f'TP {i}:', row[1][cls+0], int(tp[i].item()))
        #     print(f'FN {i}:', row[1][cls + 1], int(fn[i].item()))
        #     print(f'FP {i}:', row[1][cls + 2], int(fp[i].item()))
        #     print('')

        # === Confusion Matrix stuff
        if use_confusion_matrix:
            y_true_flat = y_true.argmax(dim=1).cpu().detach().numpy()
            y_pred_flat = torch.softmax(y_pred, dim=1).argmax(dim=1).cpu().detach().numpy()
            labels = [0, 1, 2, 3, 4] # 5 classes
            _mat = confusion_matrix(y_true_flat.flatten(), y_pred_flat.flatten(), labels=labels)
            confusion_matrix_with_bg.append(_mat)

        #=== Breakdown by image class
        # Disaster type
        if include_disaster_type_breakdown:
            for i, img_filename in enumerate(img_filenames):
                # Compute F1
                disaster_type = img_filename.split('_')[0]
                if disaster_type not in disaster_type_measurers:
                    disaster_type_measurers[disaster_type] = MultiClassF1(ignore_last_class=cfg.MODEL.BACKGROUND.TYPE == 'new-class')
                    confusion_matrices_by_disaster_type[disaster_type] = 0
                disaster_type_measurers[disaster_type].add_sample(y_true[[i]], y_pred[[i]])

                # Compute Confusion Matrix
                if use_confusion_matrix:
                    _mat = confusion_matrix(y_true_flat[i].flatten(), y_pred_flat[i].flatten(), labels=labels)
                    confusion_matrices_by_disaster_type[disaster_type] += _mat

    use_gts_mask = run_type == 'TRAIN' and cfg.DATASETS.LOCALIZATION_MASK.TRAIN_USE_GTS_MASK
    if run_type == 'TRAIN':
        dset_source = cfg.DATASETS.TRAIN[0]
        set_name = 'training_set'
    elif run_type == 'VALIDATION':
        dset_source = cfg.DATASETS.VALIDATION[0]
        set_name = 'validation_set'
    elif run_type == 'TEST':
        dset_source = cfg.DATASETS.TEST[0]
        set_name = 'test_set'

    trfm = build_transforms(cfg, use_gts_mask = use_gts_mask)
    bg_class = 'new-class' if cfg.MODEL.BACKGROUND.TYPE == 'new-class' else None
    dataset = Xview2Detectron2DamageLevelDataset(dset_source,
                                                 pre_or_post='post',
                                                 transform=trfm,
                                                 background_class=bg_class,
                                                 label_format=cfg.DATASETS.LABEL_FORMAT,)
    inference_loop(net, cfg, device, evaluate,
                   batch_size=cfg.TRAINER.BATCH_SIZE,
                   max_samples = max_samples,
                   dataset = dataset,
                   callback_include_x=True)

    # Summary gathering ===

    print('Computing F1 score ... ', end=' ', flush=True)
    # Max of the mean F1 score

    total_f1, f1_per_class = measurer.compute_f1()
    all_fpr, all_fnr = measurer.compute_basic_metrics()
    print(total_f1, flush=True)

    log_data = {f'{set_name} total F1': total_f1,
                'step': step,
                'epoch': epoch,
                }

    if include_disaster_type_breakdown:
        for disaster_type, m in disaster_type_measurers.items():
            f1 = m.compute_f1()[0]
            print(f'disaster_{disaster_type}_f1', f1)
            wandb.log({
                f'disaster-{disaster_type}-f1': f1
            }, step=global_step)
    # official Xivew2 scoring
    offical_score = XviewMetrics(allrows)
    print(f'official_f1', offical_score.df1)
    wandb.log({
        f'official-f1': offical_score.df1
    }, step=global_step)



    confmx_dir = f'{cfg.OUTPUT_DIR}/confusion_matrices/'
    if include_disaster_type_breakdown and use_confusion_matrix:
        for disaster_type, cm in confusion_matrices_by_disaster_type.items():
            name = f'disaster {disaster_type}'
            plot_confmtx(name, cm, confmx_dir)
            log_data[name] = plt

    if include_component_f1:
        loss_component_mean = np.mean(component_f1, axis=0)
        for comp_loss, cls in zip(loss_component_mean, ['no-dmg', 'minor-dmg', 'major-dmg', 'destroyed', 'bg']):
            log_data[f'{run_type}_{cls}_soft_dice'] = comp_loss

    damage_levels = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    for f1, dmg in zip(f1_per_class, damage_levels):
        log_data[f'{set_name} {dmg} f1'] = f1

    damage_levels += ['negative class']
    for fpr, fnr, dmg in zip(all_fpr, all_fnr, damage_levels):
        log_data[f'{set_name} {dmg} false negative rate'] = fnr
        log_data[f'{set_name} {dmg} false positive rate'] = fpr


    # Plot confusion matrix
    if use_confusion_matrix:
        cm = np.sum(confusion_matrix_with_bg, axis=0)
        plot_confmtx(name=run_type, cm=cm, directory=confmx_dir)
        log_data[f'confusion_matrix_{run_type}'] = plt

    wandb.log(log_data, step=global_step)

    return total_f1

def plot_confmtx(name, cm, directory):
    '''
    Plots and saves the confusion matrix into the run directory
    :param name:
    :param cm:
    :return:
    '''
    normalized_cm = cm / cm.sum(axis=-1, keepdims=True)
    labels = ['no-damage', 'minor', 'major', 'destroyed', 'background', ]

    fig, ax = plt.subplots()

    ax.matshow(normalized_cm, cmap='Blues')

    for (i, j), z in np.ndenumerate(normalized_cm):
        true_value = cm[i, j]
        ax.text(j, i, '{:0.3f}\n{}'.format(z, true_value), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax.autoscale(False)
    ax.set_yticklabels([''] + labels)
    ax.set_xticklabels([''] + labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
             rotation_mode="anchor")
    # fig.tight_layout()
    plt.title(f'{name} confusion matrix')

    # display_labels = ['no-damage', 'minor', 'major', 'destroyed', 'background', ]
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                               display_labels=display_labels)
    #
    # disp = disp.plot(include_values=True,
    #                  cmap=plt.cm.Blues, ax=None, xticks_rotation=45)

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig( path.join(directory, f'{name}.png') )

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
    # trfm.append(BGR2RGB())

    if cfg.DATASETS.LOCALIZATION_MASK.ENABLED: trfm.append(IncludeLocalizationMask(use_gts_mask))
    if cfg.DATASETS.INCLUDE_PRE_DISASTER: trfm.append(StackPreDisasterImage())
    if cfg.AUGMENTATION.RESIZE: trfm.append(Resize(scale=cfg.AUGMENTATION.RESIZE_RATIO))
    if cfg.AUGMENTATION.CROP_TYPE == 'uniform' and for_training:
        trfm.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
    elif cfg.AUGMENTATION.CROP_TYPE == 'importance' and for_training:
        trfm.append(ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE, label_type=cfg.DATASETS.LABEL_FORMAT))
    if cfg.AUGMENTATION.RANDOM_FLIP_ROTATE and for_training:
        trfm.append(RandomFlipRotate())
    trfm.append(Npy2Torch())
    if cfg.AUGMENTATION.ENABLE_VARI: trfm.append(VARI())
    if cfg.AUGMENTATION.ZERO_MEAN: trfm.append(ZeroMeanUnitImage())
    trfm = transforms.Compose(trfm)
    return trfm

def combo_loss(p, y, class_weights=None):
    y_ = y.argmax(dim=1).long()
    ce = F.cross_entropy(p, y_, weight=class_weights)
    dice = soft_dice_loss_multi_class(p, y)
    loss =  ce + dice
    # only return component loss if using weighted ce
    if class_weights is not None: return loss, (ce, dice)
    return loss

def weighted_ce_loss(p, y, class_weights=None):
    y_ = y.argmax(dim=1).long()
    loss = F.cross_entropy(p, y_, weight=class_weights)
    return loss

def image_sampling_weight(dataset_metadata):
    print('performing oversampling...', end='', flush=True)
    EMPTY_IMAGE_BASELINE = 1000
    image_p = np.array([image_desc['post']['image_weight'] for image_desc in dataset_metadata]) + EMPTY_IMAGE_BASELINE
    print('done', flush=True)
    return image_p

def class_uniform_image_sampling_weight(dataset_metadata):
    print('performing oversampling...', end='', flush=True)
    weights = []
    for image_desc in dataset_metadata:
        weights.append(image_desc['post']['image_weight_per_class'])
    weights = np.array(weights)
    class_uniform_weights = np.sum(weights / weights.sum(axis=0, keepdims=True), axis=1)
    print('done', flush=True)
    return class_uniform_weights

def gpu_stats():
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e6 # bytes to MB
    max_memory_cached = torch.cuda.max_memory_reserved() /1e6
    return int(max_memory_allocated), int(max_memory_cached)

def summarize_config(cfg, device):
    run_config = {}
    run_config['config_file'] = cfg.CONFIG
    run_config['run_name'] = cfg.NAME
    run_config['compute_node'] = cfg.COMPUTER_CONFIG
    run_config['torch_version'] = torch.__version__
    run_config['device'] = device
    run_config['log_path'] = cfg.OUTPUT_DIR
    run_config['training_set'] = cfg.DATASETS.TRAIN
    run_config['validation_set'] = cfg.DATASETS.VALIDATION
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

def mean_square_error(p, y):
    p = torch.sigmoid(p)
    return F.mse_loss(p, y)

def setup(args):
    cfg = new_config()
    cfg.COMPUTER_CONFIG = f'configs/environment/{args.computer_name}.yaml' if args.computer_name else f'configs/environment/{platform.node()}.yaml'
    if not os.path.exists(cfg.COMPUTER_CONFIG):
        print(f'No environment config for {cfg.COMPUTER_CONFIG}, reverting to default!')
        cfg.COMPUTER_CONFIG = f'configs/environment/default.yaml'
    cfg.merge_from_file(cfg.COMPUTER_CONFIG)

    cfg.merge_from_file(f'configs/damage_detection/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.CONFIG = args.config_file
    cfg.NAME = args.config_file
    cfg.resume_from = args.resume_from
    cfg.eval_only = args.eval_only
    cfg.TAGS = [cfg.CONFIG]
    cfg.PROJECT = 'urban_dl_final'

    if args.log_dir: # Override Output dir
        cfg.OUTPUT_DIR = path.join(args.log_dir, cfg.NAME)
    else:
        cfg.OUTPUT_DIR = path.join(cfg.OUTPUT_BASE_DIR, cfg.NAME)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.data_dir:
        cfg.DATASETS.TRAIN = (args.data_dir,)
    return cfg

def damage_train(trial: optuna.Trial=None, cfg=None):
    # Overwrite environment config if in debug mode
    if cfg.DEBUG:
        cfg.DATASETS.TRAIN = cfg.DATASETS.DEBUG_TRAIN
        cfg.DATASETS.VALIDATION = cfg.DATASETS.DEBUG_VALIDATION
        cfg.DATASETS.TEST = cfg.DATASETS.DEBUG_TEST
        cfg.NAME += '_debug'
        cfg.PROJECT += '_debug'

    # out_channels = cfg.MODEL.OUT_CHANNELS
    # if cfg.MODEL.BACKBONE.ENABLED:
    #     if cfg.MODEL.COMPLEX_ARCHITECTURE.ENABLED:
    #         if cfg.MODEL.COMPLEX_ARCHITECTURE.TYPE == 'pspnet':
    #             net = smp.PSPNet(cfg.MODEL.BACKBONE.TYPE,
    #                            encoder_weights=None,
    #                            encoder_depth=3,
    #                            in_channels=cfg.MODEL.IN_CHANNELS,
    #                            classes=cfg.MODEL.OUT_CHANNELS
    #                            )
    #     else:
    #         net = smp.Unet(cfg.MODEL.BACKBONE.TYPE,
    #                        encoder_weights=None,
    #                        encoder_depth=5,
    #                        decoder_channels = [512,256,128,64,32],
    #                        in_channels= cfg.MODEL.IN_CHANNELS,
    #                        classes=cfg.MODEL.OUT_CHANNELS
    #         )
    if cfg.MODEL.SIAMESE.ENABLED:
        use_pretrained = cfg.MODEL.SIAMESE.PRETRAINED
        if cfg.MODEL.SIAMESE.TYPE == 'SENET152':
            net = SeNet154_Unet_Double(use_pretrained, cfg)
        elif cfg.MODEL.SIAMESE.TYPE == 'RESNEXT50':
            net = SeResNext50_Unet_Double(use_pretrained, cfg)
        elif cfg.MODEL.SIAMESE.TYPE == 'DPN92':
            net = Dpn92_Unet_Double(use_pretrained, cfg)
        else:
            raise ValueError('Unknown simaese basenet')
    else:
        net = UNet(cfg)

    if cfg.resume_from:
        full_model_path = path.join(cfg.OUTPUT_DIR, cfg.resume_from)
        # Removing the module.** in front of keys
        filtered_dict = {}
        state_dict = torch.load(full_model_path)

        # For winning model, quick hack to remove the state_dict
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        for k, v in state_dict.items():  # ['state_dict']
            k = '.'.join(k.split('.')[1:])
            filtered_dict[k] = v
        net.load_state_dict(filtered_dict)
        print('Model loaded from {}'.format(full_model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cudnn.benchmark = True # faster convolutions, but more memory

    print('=== Runnning on device: p', device)

    objective = 0.0

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        if cfg.eval_only:
            wandb.init(
                name=cfg.NAME,
                project=cfg.PROJECT,
                entity='eoai4globalchange',
                tags=cfg.TAGS + ['eval', 'dmg'],
                config=cfg,
            )
            dmg_model_eval(net, cfg, device, run_type='TEST', max_samples=None, use_confusion_matrix=True,
                           include_disaster_type_breakdown=True)
            dmg_model_eval(net, cfg, device, run_type='VALIDATION', max_samples=None, use_confusion_matrix=True,
                           include_disaster_type_breakdown=True)
            dmg_model_eval(net, cfg, device, run_type='TRAIN', max_samples=1000, use_confusion_matrix=True,
                           include_disaster_type_breakdown=True)
        else:
            # Dynamically adjust the batch size to fit on GPU
            orginal_batch_size = cfg.TRAINER.BATCH_SIZE
            id = wandb.util.generate_id()
            while(cfg.TRAINER.BATCH_SIZE):
                try:
                    wandb.init(
                        id=id,
                        name=cfg.NAME,
                        project=cfg.PROJECT,
                        entity='eoai4globalchange',
                        tags=cfg.TAGS + ['train', 'dmg'],
                        config=cfg,
                        reinit=True,
                        resume='allow'
                    )
                    objective = train_net(net, cfg, device, trial)
                    break
                except RuntimeError as runerr:
                    # Check if runtime error is CUDA error
                    if 'CUDA out of memory' not in str(runerr):
                        raise runerr
                    cfg.TRAINER.BATCH_SIZE = cfg.TRAINER.BATCH_SIZE - 2
                    print("Run Time Error:" + str(runerr), file=sys.stderr)
                    print(f'!!!! ---> Original batch size ({orginal_batch_size}) too large, trying batch size: {cfg.TRAINER.BATCH_SIZE}',
                          file=sys.stderr)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    # Return an objective measurement so this can be used in a hyperparam optimizer
    return objective

if __name__ == '__main__':
    args = default_argument_parser().parse_known_args()[0]
    cfg = setup(args)
    damage_train(None, cfg)