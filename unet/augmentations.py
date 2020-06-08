import os

import torchvision.transforms.functional as TF

import cv2
import numpy as np
import torch
from scipy import ndimage
from unet.utils.utils import *

class Resize():

    def __init__(self, scale, resize_label=True):
        self.scale = scale
        self.resize_label = resize_label

    def __call__(self, args):
        input, label, image_path = args

        input = cv2.resize(input, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if self.resize_label:
            label = cv2.resize(label, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        return input, label, image_path

class VARI():
    def __call__(self, args):
        input, label, image_path = args
        image_name = os.path.basename(image_path)
        dir_name = os.path.dirname(image_path)
        vari_path = os.path.join(dir_name, 'clahe_vari' ,image_name)
        if os.path.exists(vari_path):
            mask = imread_cached(vari_path).astype(np.float32)[...,[0]]
            input_t = np.concatenate([input, mask], axis=-1)
            return input_t, label, image_path
        # Input is in BGR
        assert input.shape[1] == input.shape[2] and torch.is_tensor(input), 'invalid tensor, did you forget to put VARI after Np2Torch?'
        R = input[0]
        G = input[1]
        B = input[2]
        eps = 1e-6
        VARI = (G-R) / 2 * (eps + G+R-B) + 0.5 # Linearly transformed to be [0, 1]
        VARI = VARI.unsqueeze(0)
        input_t = torch.cat([input, VARI])
        return input_t, label, image_path

class Npy2Torch():
    def __call__(self, args):
        input, label, image_path = args
        input_t = TF.to_tensor(input)
        label = TF.to_tensor(label)
        return input_t, label, image_path
class BGR2RGB():
    def __call__(self, args):
        input, label, image_path = args
        input = bgr2rgb(input)
        return input, label, image_path

class ZeroMeanUnitImage():
    def __call__(self, args):
        input, label, image_path = args
        input /= 128
        input -= 1
        return input, label, image_path

class UniformCrop():
    '''
    Performs uniform cropping on numpy images (cv2 images)
    '''
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def random_crop(self, input, label):
        image_size = input.shape[-2]
        crop_limit = image_size - self.crop_size
        x, y = np.random.randint(0, crop_limit, size=2)

        input = input[y:y+self.crop_size, x:x+self.crop_size, :]
        label = label[y:y+self.crop_size, x:x+self.crop_size]
        return input, label

    def __call__(self, args):
        input, label, image_path = args
        input, label = self.random_crop(input, label)
        return input, label, image_path

class ImportanceRandomCrop(UniformCrop):
    def __call__(self, args):
        input, label, image_path = args

        SAMPLE_SIZE = 5 # an arbitrary number that I came up with
        BALANCING_FACTOR = 200

        random_crops = [self.random_crop(input, label) for i in range(SAMPLE_SIZE)]
        # TODO Multi class vs edge mask
        weights = []
        for input, label in random_crops:
            if label.shape[2] >= 4:
                # Damage detection, multi class, excluding backround
                weights.append(label[...,:-1].sum())
            elif label.shape[2] > 1:
                # Edge Mask, excluding edge masks
                weights.append(label[...,0].sum())
            else:
                weights.append(label.sum())
        crop_weights = np.array([label.sum() for input, label in random_crops]) + BALANCING_FACTOR
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(SAMPLE_SIZE, p=crop_weights)
        input, label = random_crops[sample_idx]

        return input, label, image_path

class IncludeLocalizationMask():
    def __init__(self, use_gts_mask=False):
        self.use_gts_mask = use_gts_mask

    def __call__(self, args):
        input, label, image_path = args
        image_name = os.path.basename(image_path)

        # Load predisaster counter part
        img_name_split = image_name.split('_')
        img_name_split[-2] = 'pre'
        image_name = '_'.join(img_name_split)

        dir_name = os.path.dirname(image_path)
        # Load preprocessed mask if exist
        subdir = 'label_mask' if self.use_gts_mask else 'loc_predicted'
        mask_path = os.path.join(dir_name, subdir, image_name)

        assert os.path.exists(mask_path), 'Mask data is not generated, please double check \n' + mask_path

        mask = imread_cached(mask_path).astype(np.float32)
        mask = mask[...,0][...,None] # [H, W, 3] -> [H, W, 1]

        input = np.concatenate([input, mask], axis=-1)

        return input, label, image_path
class StackPreDisasterImage():
    def __init__(self, pre_or_post='pre'):
        self.pre_or_post = pre_or_post

    def __call__(self, args):
        input, label, image_path = args
        image_name = os.path.basename(image_path)
        dir_name = os.path.dirname(image_path)

        # Load counter part
        img_name_split = image_name.split('_')
        img_name_split[-2] = self.pre_or_post
        cp_image_name = '_'.join(img_name_split)

        # Read image
        cp_image_path = os.path.join(dir_name, cp_image_name)
        cp_image = imread_cached(cp_image_path).astype(np.float32)

        # RGB -> BGR and stack
        # cp_image = bgr2rgb(cp_image)
        input = np.concatenate([input, cp_image], axis=-1)
        return input, label, image_path

class RandomFlipRotate():
    def __call__(self, args):
        input, label, image_path = args
        _hflip = np.random.choice([True, False])
        _vflip = np.random.choice([True, False])
        _rot = np.random.randint(0, 360)

        if _hflip:
            input = np.flip(input, axis=0)
            label = np.flip(label, axis=0)

        if _vflip:
            input = np.flip(input, axis=1)
            label = np.flip(label, axis=1)

        # input = ndimage.rotate(input, _rot, reshape=False).copy()
        # label = ndimage.rotate(label, _rot, reshape=False).copy()
        input = input.copy()
        label = label.copy()
        return input, label, image_path

def bgr2rgb(img):
    return img[..., [2,1,0]]