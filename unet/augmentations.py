import os

import torchvision.transforms.functional as TF

import cv2
import numpy as np
import torch
from scipy import ndimage
from unet.utils.utils import *


class Resize:

    def __init__(self, scale, resize_label=True):
        self.scale = scale
        self.resize_label = resize_label

    def __call__(self, args):
        args['x'] = cv2.resize(args['x'], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.resize_label:
            args['y'] = cv2.resize(args['y'], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        return args


class Dilate:
    def __init__(self, kernel=np.ones((3,3)), iterations=2):
        self.kernel = kernel
        self.iterations = iterations

    def __call__(self, args):
        # args['y'] = cv2.dilate(args['y'].T, kernel=self.kernel, iterations=self.iterations).T
        # Above needs a retraded workaround because cv2
        for i in range(args['y'].shape[-1]):
            args['y'][..., i]= cv2.dilate(args['y'][..., i], kernel=self.kernel, iterations=self.iterations)
        return args


class VARI:
    def __call__(self, args):
        image_name = os.path.basename(args['img_path'])
        dir_name = os.path.dirname(args['img_path'])
        vari_path = os.path.join(dir_name, 'clahe_vari', image_name)
        if os.path.exists(vari_path):
            mask = imread_cached(vari_path).astype(np.float32)[...,[0]]
            args['x'] = np.concatenate([args['x'], mask], axis=-1)
            return args
        assert args['x'].shape[1] == args['x'].shape[2], 'Invalid tensor'
        assert torch.is_tensor(args['x']), 'Not a tensor, put VARI after NumpyToTorch'
        # args['x'] is in BGR
        R = args['x'][0]
        G = args['x'][1]
        B = args['x'][2]
        eps = 1e-6
        VARI = (G-R) / 2 * (eps + G+R-B) + 0.5 # Linearly transformed to be [0, 1]
        VARI = VARI.unsqueeze(0)
        args['x'] = torch.cat([args['x'], VARI])
        return args


class Npy2Torch:
    def __call__(self, args):
        args['x'] = TF.to_tensor(args['x'])
        args['y'] = TF.to_tensor(args['y'])
        args['loc_mask_pred'] = TF.to_tensor(args['loc_mask_pred'])
        args['loc_mask_gt'] = TF.to_tensor(args['loc_mask_gt'])
        return args


class BGR2RGB:
    def __call__(self, args):
        args['x'] = bgr2rgb(args['x'])
        return args


class ZeroMeanUnitImage:
    def __call__(self, args):
        args['x'] /= 128
        args['x'] -= 1
        return args


class UniformCrop:
    '''
    Performs uniform cropping on numpy images (cv2 images)
    '''
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def random_crop(self, image, label):
        image_size = image.shape[-2]
        crop_limit = image_size - self.crop_size
        x, y = np.random.randint(0, crop_limit, size=2)

        input = image[y:y+self.crop_size, x:x+self.crop_size, ...]
        label = label[y:y+self.crop_size, x:x+self.crop_size, ...]
        return input, label

    def __call__(self, args):
        args['x'], args['y'] = self.random_crop(args['x'], args['y'])
        return args


class ImportanceRandomCrop(UniformCrop):
    def __init__(self, crop_size, label_type):
        super().__init__(crop_size)
        self.label_type = label_type

    def __call__(self, args):
        SAMPLE_SIZE = 5  # an arbitrary number that I came up with
        BALANCING_FACTOR = 200

        random_crops = [self.random_crop(args['x'], args['y']) for i in range(SAMPLE_SIZE)]
        # TODO Multi class vs edge mask
        weights = []
        for args['x'], args['y'] in random_crops:
            if self.label_type == 'ordinal':
                weights.append(args['y'][...,-1].sum())
            elif args['y'].shape[2] >= 4:
                # Damage detection, multi class, excluding backround
                weights.append(args['y'][...,:-1].sum())
            elif args['y'].shape[2] > 1:
                # Edge Mask, excluding edge masks
                weights.append(args['y'][...,0].sum())
            else:
                weights.append(args['y'].sum())
        crop_weights = np.array([args['y'].sum() for args['x'], args['y'] in random_crops]) + BALANCING_FACTOR
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(SAMPLE_SIZE, p=crop_weights)
        args['x'], args['y'] = random_crops[sample_idx]

        return args


class IncludeLocalizationMask:
    def __init__(self, use_gts_mask=False):
        self.use_gts_mask = use_gts_mask

    def __call__(self, args):
        args['x'] = np.concatenate([args['x'], args['loc_mask_pred']], axis=-1)
        return args


class StackPreDisasterImage:
    def __init__(self, pre_or_post='pre'):
        self.pre_or_post = pre_or_post

    def __call__(self, args):
        image_name = os.path.basename(args['img_path'])
        dir_name = os.path.dirname(args['img_path'])

        # Load counter part
        img_name_split = image_name.split('_')
        img_name_split[-2] = self.pre_or_post
        cp_image_name = '_'.join(img_name_split)

        # Read image
        cp_image_path = os.path.join(dir_name, cp_image_name)
        cp_image = imread_cached(cp_image_path).astype(np.float32)

        # RGB -> BGR and stack
        # cp_image = bgr2rgb(cp_image)
        args['x'] = np.concatenate([args['x'], cp_image], axis=-1)
        return args


class RandomFlipRotate:
    def __call__(self, args):
        _hflip = np.random.choice([True, False])
        _vflip = np.random.choice([True, False])
        _rot = np.random.randint(0, 360)

        if _hflip:
            args['x'] = np.flip(args['x'], axis=0)
            args['y'] = np.flip(args['y'], axis=0)

        if _vflip:
            args['x'] = np.flip(args['x'], axis=1)
            args['y'] = np.flip(args['y'], axis=1)

        # args['x'] = ndimage.rotate(args['x'], _rot, reshape=False).copy()
        # args['y'] = ndimage.rotate(args['y'], _rot, reshape=False).copy()
        args['x'] = args['x'].copy()
        args['y'] = args['y'].copy()
        return args


def bgr2rgb(img):
    return img[..., [2,1,0]]