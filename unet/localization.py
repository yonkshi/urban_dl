import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils import data as torch_data
from torchvision import transforms, utils
from torch import optim

from unet import UNet
from unet.dataloader import Xview2Detectron2Dataset
from unet.augmentations import *


class LocalizationModel(pl.LightningModule):
    def __init__(self, net:nn.Module, loss, cfg):
        super().__init__()
        self.net = net
        self.loss = loss
        self.cfg = cfg

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        pred = self.forward(x)
        # TODO Include Edge Loss
        return {'loss': self.loss(pred, y)}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x = batch['x']
        y = batch['y']
        pred = self.forward(x)

        return {'val_loss': self.loss(pred, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    def configure_optimizers(self):
        # REQUIRED
        return optim.Adam(self.net.parameters(),
                          lr=self.cfg.TRAINER.LR,
                          weight_decay=0.0005)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        cfg = self.cfg
        use_edge_loss = cfg.MODEL.LOSS_TYPE == 'FrankensteinEdgeLoss'
        trfm = []
        trfm.append(BGR2RGB())
        if cfg.DATASETS.USE_CLAHE_VARI: trfm.append(VARI())
        if cfg.AUGMENTATION.RESIZE: trfm.append(Resize(scale=cfg.AUGMENTATION.RESIZE_RATIO))
        if cfg.AUGMENTATION.CROP_TYPE == 'uniform':
            trfm.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
        elif cfg.AUGMENTATION.CROP_TYPE == 'importance':
            trfm.append(ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
        if cfg.AUGMENTATION.RANDOM_FLIP_ROTATE: trfm.append(RandomFlipRotate())

        trfm.append(Npy2Torch())
        trfm = transforms.Compose(trfm)

        dataset = Xview2Detectron2Dataset(cfg.DATASETS.TRAIN[0],
                                          pre_or_post=cfg.DATASETS.PRE_OR_POST,
                                          include_image_weight=True,
                                          transform=trfm,
                                          include_edge_mask=use_edge_loss,
                                          use_clahe=cfg.DATASETS.USE_CLAHE_VARI,

                                          )


        dataloader_kwargs = {
            'batch_size': cfg.TRAINER.BATCH_SIZE,
            'num_workers': cfg.DATALOADER.NUM_WORKER,
            'shuffle': cfg.DATALOADER.SHUFFLE,
            'drop_last': True,
            'pin_memory': True,
        }
        # sampler
        if cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'simple':
            image_p = self.image_sampling_weight(dataset.dataset_metadata)
            sampler = torch_data.WeightedRandomSampler(weights=image_p, num_samples=len(image_p))
            dataloader_kwargs['sampler'] = sampler
            dataloader_kwargs['shuffle'] = False
        dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        # REQUIRED
        cfg = self.cfg
        use_edge_loss = cfg.MODEL.LOSS_TYPE == 'FrankensteinEdgeLoss'
        trfm = []
        trfm.append(BGR2RGB())
        if cfg.DATASETS.USE_CLAHE_VARI: trfm.append(VARI())
        if cfg.AUGMENTATION.RESIZE: trfm.append(Resize(scale=cfg.AUGMENTATION.RESIZE_RATIO))
        if cfg.AUGMENTATION.CROP_TYPE == 'uniform':
            trfm.append(UniformCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
        elif cfg.AUGMENTATION.CROP_TYPE == 'importance':
            trfm.append(ImportanceRandomCrop(crop_size=cfg.AUGMENTATION.CROP_SIZE))
        if cfg.AUGMENTATION.RANDOM_FLIP_ROTATE: trfm.append(RandomFlipRotate())

        trfm.append(Npy2Torch())
        trfm = transforms.Compose(trfm)

        dataset = Xview2Detectron2Dataset(cfg.DATASETS.TRAIN[0],
                                          pre_or_post=cfg.DATASETS.PRE_OR_POST,
                                          include_image_weight=True,
                                          transform=trfm,
                                          include_edge_mask=use_edge_loss,
                                          use_clahe=cfg.DATASETS.USE_CLAHE_VARI,

                                          )

        dataloader_kwargs = {
            'batch_size': cfg.TRAINER.BATCH_SIZE,
            'num_workers': cfg.DATALOADER.NUM_WORKER,
            'shuffle': cfg.DATALOADER.SHUFFLE,
            'drop_last': True,
            'pin_memory': True,
        }
        # sampler
        if cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE == 'simple':
            image_p = self.image_sampling_weight(dataset.dataset_metadata)
            sampler = torch_data.WeightedRandomSampler(weights=image_p, num_samples=len(image_p))
            dataloader_kwargs['sampler'] = sampler
            dataloader_kwargs['shuffle'] = False
        dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
        return dataloader

    def image_sampling_weight(self, dataset_metadata):
        print('performing oversampling...', end='', flush=True)
        EMPTY_IMAGE_BASELINE = 1000
        image_p = np.array(
            [image_desc['pre']['image_weight'] for image_desc in dataset_metadata]) + EMPTY_IMAGE_BASELINE
        print('done', flush=True)
        # normalize to [0., 1.]
        image_p = image_p
        return image_p
