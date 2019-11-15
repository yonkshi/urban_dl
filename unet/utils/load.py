#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import torch
import h5py
import json
import cv2
from unet.utils import *
from debug_tools import __benchmark_init, benchmark
import pycocotools.mask as mask_utils

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

class Xview2Detectron2Dataset(torch.utils.data.Dataset):
    '''
    Dataset for Detectron2 style labelled Dataset
    '''
    def __init__(self, file_path, timeidx):
        super().__init__()

        ds_path = os.path.join(file_path,'labels.json')
        with open(ds_path) as f:
            ds = json.load(f)
        self.dataset = ds
        self.dataset_path = file_path

        self.length = len(ds)
        self.timeidx = timeidx


        self.label_mask_cache = {}

    def __getitem__(self, index):
        data_sample = self.dataset[index]
        input, image_shape =self._process_input(data_sample['file_name'])
        label = self._extract_label(data_sample['annotations'], image_shape)
        # label = label[None, ...] # C x H x W
        sample_name = data_sample['file_name']

        return input, label, sample_name

    def polygons_to_bitmask(self, polygons, height, width) -> np.ndarray:
        """
        Args:
            polygons (list[ndarray]): each array has shape (Nx2,)
            height, width (int)
        Returns:
            ndarray: a bool mask of shape (height, width)
        """
        assert len(polygons) > 0, "COCOAPI does not support empty polygons"
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        return mask_utils.decode(rle).astype(np.bool)


    def _extract_label(self, annotations_set, image_size):
        masks = []
        for anno in annotations_set:
            segm_instances = anno['segmentation'][0]

            # Converting from XYXY to [[x,y],[x,y]
            num_polygon_points = len(segm_instances) / 2
            assert num_polygon_points.is_integer(), 'The polygon array must be in XYXY format'
            polygon_pts = np.reshape(segm_instances, (int(num_polygon_points), 2))
            mask = self.polygons_to_bitmask(polygon_pts, image_size[0], image_size[1])
            masks.append(mask)

        if masks:
            composite_mask = np.any(masks, axis=0).astype(np.long)
        else:
            composite_mask = np.zeros(image_size, dtype=np.long)
        return composite_mask

    def _process_input(self, image_filename):
        img_path = os.path.join(self.dataset_path, image_filename)
        img = cv2.imread(img_path)
        # BGR to RGB
        img = img[...,::-1]

        input = img.astype(np.float32) / 255.
        # move from (x, y, c) to (c, x, y) PyTorch style
        input = np.moveaxis(input, -1, 0)
        image_shape = input.shape[1:]
        return input, image_shape

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
