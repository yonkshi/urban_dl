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
    def __init__(self, file_path, cfg, random_crop, resize_label=True,  include_index=False, transform = None,):
        super().__init__()

        ds_path = os.path.join(file_path,'labels.json')
        with open(ds_path) as f:
            ds = json.load(f)
        self.dataset = ds
        self.dataset_path = file_path

        self.length = len(ds)
        print('dataset length', self.length)
        self._cfg = cfg
        self.include_index = include_index
        self._should_random_crop = random_crop
        self._should_resize_label = resize_label
        self.label_mask_cache = {}
        self.transform = transform

    def __getitem__(self, index):
        data_sample = self.dataset[index]
        input, image_shape =self._process_input(data_sample['file_name'])
        label = self._extract_label(data_sample['annotations'], image_shape)
        # label = label[None, ...] # C x H x W
        sample_name = data_sample['file_name']

        if self._cfg.AUGMENTATION.RESIZE and self._should_resize_label:
            # Resize label
            scale = self._cfg.AUGMENTATION.RESIZE_RATIO
            label = cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if self._should_random_crop:
            input, label = self._random_crop(input, label)

        if self.transform:
            input, label = self.transform([input, label])

        if self.include_index:
            return input, label, sample_name, index
        return input, label, sample_name

    def _extract_label(self, annotations_set, image_size):
        masks = []

        building_polygons = []
        for anno in annotations_set:
            building_polygon_xy = np.array(anno['segmentation'][0], dtype=np.int32).reshape(-1, 2)
            building_polygons.append(building_polygon_xy)

        mask2 = np.zeros((1024, 1024), dtype=np.uint8)
        cv2.fillPoly(mask2, building_polygons, 1)
        mask = mask2.astype(np.float32)

        return mask

    def _random_crop(self, input, label):

        assert input.shape[-1] == input.shape[-2], 'Image must be square shaped, or did you rotate the axis wrong? '
        crop_size = self._cfg.AUGMENTATION.CROP_SIZE
        image_size = input.shape[-1]
        crop_limit = image_size - crop_size
        crop_x, crop_y = np.random.randint(0, crop_limit, 2)

        cropped_input = input[:, crop_y:crop_size + crop_y, crop_x:crop_size+crop_x]
        cropped_label = label[crop_y:crop_size + crop_y, crop_x:crop_size+crop_x]

        return cropped_input, cropped_label

    def _process_input(self, image_filename):
        img_path = os.path.join(self.dataset_path, image_filename)
        img = cv2.imread(img_path)

        if self._cfg.AUGMENTATION.RESIZE:
            # Resize image
            scale = self._cfg.AUGMENTATION.RESIZE_RATIO
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # BGR to RGB
        img = img[...,::-1]

        input = img.astype(np.float32) / 255.
        # move from (x, y, c) to (c, x, y) PyTorch style
        input = np.moveaxis(input, -1, 0)
        image_shape = input.shape[1:]
        return input, image_shape

    def __len__(self):

        return self.length

class SloveniaDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, timeidx):
        super().__init__()
        self.dataset = h5py.File(file_path, 'r', libver='latest', swmr=True)
        self.dataset_indices = list(self.dataset.keys())
        self.episode = None

        self.length = 0 #len(list(self.dataset.keys()))
        self.timeidx = timeidx
        self._precompute_comp()

        self.label_mask_cache = {}

    def __getitem__(self, index):

        dset_idx, time_idx = self.random_dset_indices[index]
        subset_name = self.dataset_indices[dset_idx]
        subset = self.dataset[subset_name]
        dset = subset['data_bands']
        dset.refresh()
        # timeidx = self.timeidx % self.length # circular time step
        obs = dset[time_idx]
        # move from (x, y, c) to (c, x, y) PyTorch style
        obs = np.moveaxis(obs, -1, 0)
        # sometimes data can exceed [0, 1], clip em!
        obs = np.clip(obs, 0,1)

        cloud_mask = subset['mask/valid_data'][time_idx].astype(np.float32)
        cloud_mask = np.moveaxis(cloud_mask, -1, 0)

        label = subset['mask_timeless']['lulc'][...,0]
        label = label.astype(np.long)

        # check if label no datamask has been cached
        if subset_name not in self.label_mask_cache.keys():
            mask = (label != 0).astype(np.float32)
            self.label_mask_cache[subset_name] = mask
        label_nodata_mask = self.label_mask_cache[subset_name]
        label_nodata_mask = label_nodata_mask[None,...] # add a dim to match obs

        sample_name = f'{subset_name}, t={time_idx}'

        # label.refresh()
        # label = label1.value
        # label = np.moveaxis(label, -1, 0).squeeze().astype(np.long)
        # label = np.argmax(label, axis=0)
        return obs, label, cloud_mask, label_nodata_mask, sample_name

    def _precompute_comp(self):

        print('precomputing...')
        # computing total length of the data
        dset_indices = []
        for dset_index, key in enumerate(self.dataset_indices):
            ts = self.dataset[f'{key}/timestamp']
            n_time_steps = ts.shape[0]
            self.length += n_time_steps

            #
            arange_timesteps = np.arange(n_time_steps)
            arange_index = [dset_index] * n_time_steps
            tuples = np.vstack((arange_index, arange_timesteps))
            dset_indices.append(tuples)

        self.random_dset_indices = np.concatenate(dset_indices, axis=-1).T
        np.random.shuffle(self.random_dset_indices)

        print('precompute complete')


    def __len__(self):

        return self.length

class Xview2Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, timeidx):
        super().__init__()
        self.dataset = h5py.File(file_path, 'r', libver='latest', swmr=True)
        self.dataset_indices = list(self.dataset.keys())
        self.episode = None

        self.length = 0 #len(list(self.dataset.keys()))
        self.timeidx = timeidx
        self._precompute_comp()

        self.label_mask_cache = {}

    def __getitem__(self, index):

        dset_idx, img_idx = self.random_dset_indices[index]
        subset_name = self.dataset_indices[dset_idx]
        subset = self.dataset[subset_name]

        input = subset['pre'][img_idx]
        input = self._process_input(input)

        label = subset['labels'][img_idx]
        label = self._process_label(label)

        sample_name = f'{subset_name}, t={img_idx}'

        # test set
        val_idx = index % len(self.random_valset_indices)
        val_dset_idx, val_img_idx = self.random_valset_indices[val_idx]
        subset_name = self.dataset_indices[val_dset_idx]
        subset = self.dataset[subset_name]

        input_val = subset['pre'][val_img_idx]
        input_val = self._process_input(input_val)

        label_val = subset['labels'][val_img_idx]
        label_val = self._process_label(label_val)
        sample_name_val = f'{subset_name}, t={val_img_idx}'

        return input, label, sample_name, input_val, label_val, sample_name_val

    def _process_label(self, label):
        label = label.sum(axis=0, keepdims=True, )
        background_layer = np.ones_like(label[0])[None, ...] * 0.5
        label = np.vstack([background_layer, label])
        label = np.argmax(label, axis=0)
        label = label.astype(np.long)
        return label

    def _process_input(self, input):
        input = input.astype(np.float32) / 255.
        # move from (x, y, c) to (c, x, y) PyTorch style
        input = np.moveaxis(input, -1, 0)
        return input

    def get_fixed_validation_data(self,):
        pass

    def _precompute_comp(self):

        print('precomputing...')
        # computing total length of the data
        dset_indices = []
        dset_val_indices = []
        for dset_index, key in enumerate(self.dataset_indices):
            ts = self.dataset[f'{key}/pre']
            n_images = ts.shape[0]

            # splitting training with validation set
            n_train = int(np.floor(n_images * .9))
            n_val = n_images - n_train
            self.length += n_train

            # training set
            arange_images = np.arange(n_train)
            arange_index = [dset_index] * n_train
            tuples = np.vstack((arange_index, arange_images))
            dset_indices.append(tuples)

            # validation set
            arange_images_val = np.arange(n_train, n_images)
            arange_index_val = [dset_index] * n_val
            tuples = np.vstack((arange_index_val, arange_images_val))
            dset_val_indices.append(tuples)



        self.random_dset_indices = np.concatenate(dset_indices, axis=-1).T
        self.random_valset_indices = np.concatenate(dset_val_indices, axis=-1).T
        np.random.shuffle(self.random_dset_indices)
        np.random.shuffle(self.random_valset_indices)
        print('dataset size:', self.length)
        print('precompute complete')


    def __len__(self):

        return self.length

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
