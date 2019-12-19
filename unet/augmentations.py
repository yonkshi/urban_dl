import os

import torchvision.transforms.functional as TF
import cv2
import numpy as np
import torch

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

        # TODO load existing VARI chanel
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
        if len(label.shape) > 2:
            # Label is multi class
            label = TF.to_tensor(label)
        return input_t, label, image_path
class BGR2RGB():
    def __call__(self, args):
        input, label, image_path = args
        input = input[..., [2,1,0]]
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
        crop_weights = np.array([label.sum() for input, label in random_crops]) + BALANCING_FACTOR
        crop_weights = crop_weights / crop_weights.sum()

        sample_idx = np.random.choice(SAMPLE_SIZE, p=crop_weights)
        input, label = random_crops[sample_idx]

        return input, label, image_path

class IncludeLocalizationMask():
    def __call__(self, args):
        input, label, image_path = args
        image_name = os.path.basename(image_path)
        dir_name = os.path.dirname(image_path)
        # Load preprocessed mask if exist
        mask_path = os.path.join(dir_name, 'label_mask',image_name)

        assert os.path.exists(mask_path), 'Mask data is not generated, please double check'

        mask = cv2.imread(mask_path).astype(np.float32)
        mask = mask[...,0][...,None] # [H, W, 3] -> [H, W, 1]

        input = np.concatenate([input, mask], axis=-1)

        return input, label, image_path