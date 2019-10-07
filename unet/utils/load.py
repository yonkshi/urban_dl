#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import torch
import h5py

from unet.utils import *
from debug_tools import __benchmark_init, benchmark


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
        dset = subset['pre']
        obs = dset[img_idx].astype(np.float32) / 255.
        # move from (x, y, c) to (c, x, y) PyTorch style
        obs = np.moveaxis(obs, -1, 0)

        label_raw = subset['labels'][img_idx]
        label_raw = label_raw.sum(axis=0, keepdims=True, )
        background_layer = np.ones_like(label_raw[0])[None, ...] * 0.5
        label_raw = np.vstack([background_layer, label_raw])
        label = np.argmax(label_raw, axis=0)
        label = label.astype(np.long)

        sample_name = f'{subset_name}, t={img_idx}'

        return obs, label, sample_name

    def _precompute_comp(self):

        print('precomputing...')
        # computing total length of the data
        dset_indices = []
        for dset_index, key in enumerate(self.dataset_indices):
            ts = self.dataset[f'{key}/pre']
            n_images = ts.shape[0]
            self.length += n_images

            #
            arange_images = np.arange(n_images)
            arange_index = [dset_index] * n_images
            tuples = np.vstack((arange_index, arange_images))
            dset_indices.append(tuples)

        self.random_dset_indices = np.concatenate(dset_indices, axis=-1).T
        np.random.shuffle(self.random_dset_indices)

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
