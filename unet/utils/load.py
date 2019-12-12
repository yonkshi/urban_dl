#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import torch
import h5py
import json
from PIL import Image, ImageDraw
import cv2
from unet.utils import *
from debug_tools import __benchmark_init, benchmark
import pycocotools.mask as mask_utils

class Xview2Detectron2Dataset(torch.utils.data.Dataset):
    '''
    Dataset for Detectron2 style labelled Dataset
    '''
    def __init__(self, file_path,
                 include_index=False,
                 image_oversampling = None,
                 include_image_weight = False,
                 transform = None,
                 legacy_mask_rasterization=False):
        super().__init__()

        ds_path = os.path.join(file_path,'labels.json')
        with open(ds_path) as f:
            ds = json.load(f)
        self.dataset = ds
        self.dataset_path = file_path

        self.length = len(ds)
        print('dataset length', self.length)
        # self._cfg = cfg
        self.include_index = include_index
        # self._crop_type = crop_type
        self.oversampling = image_oversampling
        self.label_mask_cache = {}
        self.include_image_weight = include_image_weight
        self.transform = transform
        self._preprocessing()
        self.legacy_mask_rasterization = legacy_mask_rasterization

    def __getitem__(self, index):

        if self.oversampling == 'simple':
            idx = np.random.choice(np.arange(0, self.length), p=self.image_p)
            data_sample = self.dataset[idx]
        else:
            data_sample = self.dataset[index]

        input, image_shape =self._process_input(data_sample['file_name'])
        label = self._extract_label(data_sample['annotations'], image_shape)
        # label = label[None, ...] # C x H x W
        sample_name = data_sample['file_name']

        # if self._cfg.AUGMENTATION.RESIZE and self._should_resize_label:
        #     # Resize label
        #     scale = self._cfg.AUGMENTATION.RESIZE_RATIO
        #     label = cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # input, label = self._random_crop(input, label, data_sample)

        if self.transform:
            input, label = self.transform([input, label])

        ret = [input, label, sample_name]
        if self.include_index:
            ret += [index]
        if self.include_image_weight:
            # Used for oversampling stats
            ret += [data_sample['image_weight']]


        return ret

    def _extract_label(self, annotations_set, image_size):
        if self.legacy_mask_rasterization:
            mask = Image.new('L', (1024, 1024), 0)
            for anno in annotations_set:
                building_polygon = anno['segmentation'][0]
                ImageDraw.Draw(mask).polygon(building_polygon, outline=1, fill=1)

            mask = np.asarray(mask).astype(np.float32)
            return mask

        else:
            building_polygons = []
            for anno in annotations_set:
                building_polygon_xy = np.array(anno['segmentation'][0], dtype=np.int32).reshape(-1, 2)
                building_polygons.append(building_polygon_xy)

            mask2 = np.zeros((1024, 1024), dtype=np.uint8)
            cv2.fillPoly(mask2, building_polygons, 1)
            mask = mask2.astype(np.float32)
            return mask

    def _random_crop(self, input, label, img_metadata):
        assert input.shape[-1] == input.shape[-2], 'Image must be square shaped, or did you rotate the axis wrong? '
        crop_size = round(self._cfg.AUGMENTATION.CROP_SIZE / 2)
        image_size = input.shape[-1]
        crop_limit = image_size - crop_size
        if self._crop_type == 'none':
            return input, label
        elif self._crop_type == 'uniform':

            crop_x, crop_y = np.random.randint(crop_size, crop_limit, 2)
        elif self._crop_type == 'gaussian':
            # background uniform random, in case gaussian went out of border
            crop_x_uni, crop_y_uni = np.random.randint(crop_size, crop_limit, 2)

            image_weight = img_metadata['image_weight']
            sigma = np.sqrt(image_weight)
            mu = img_metadata['image_com']
            # ratio = image_weight / (1024 ** 2)
            # ttt = (np.random.randn(100, 2) * sigma + mu).round().astype(np.int32)
            crop_y, crop_x = (np.random.randn(2) * sigma + mu).round().astype(np.int32)

            # use random crop if sample is out of bound or image is empty (sigma == 0)
            crop_x = crop_x if crop_size < crop_x < crop_limit and sigma != 0 else crop_x_uni
            crop_y = crop_y if crop_size < crop_y < crop_limit and sigma != 0 else crop_y_uni


        cropped_input = input[:, crop_y - crop_size: crop_y + crop_size, crop_x - crop_size: crop_x + crop_size]
        cropped_label = label[crop_y - crop_size: crop_y + crop_size, crop_x - crop_size: crop_x + crop_size]
        return cropped_input, cropped_label

    def _process_input(self, image_filename):
        img_path = os.path.join(self.dataset_path, image_filename)
        img = cv2.imread(img_path)

        # if self._cfg.AUGMENTATION.RESIZE:
        #     # Resize image
        #     scale = self._cfg.AUGMENTATION.RESIZE_RATIO
        #     img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # BGR to RGB
        # img = img[...,::-1]

        # input = img.astype(np.float32) / 255.
        # move from (x, y, c) to (c, x, y) PyTorch style
        # input = np.moveaxis(input, -1, 0)
        image_shape = img.shape[1:3]
        return img, image_shape

    def simple_oversampling_preprocess(self):
        print('performing oversampling...', end='', flush=True)
        EMPTY_IMAGE_BASELINE = 1000
        image_p = np.array([image_desc['image_weight'] for image_desc in self.dataset]) + EMPTY_IMAGE_BASELINE
        print('done', flush=True)
        # normalize to [0., 1.]
        image_p = image_p / image_p.sum()
        self.image_p = image_p
    def _preprocessing(self):
        if self.oversampling is not None and self.oversampling != 'none':
            self.simple_oversampling_preprocess()

    def __len__(self):
        return self.length
