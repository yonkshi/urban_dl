#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import torch
import json
from unet.utils import *



class Xview2Detectron2Dataset(torch.utils.data.Dataset):
    '''
    Dataset for Detectron2 style labelled Dataset
    '''
    def __init__(self, file_path,
                 pre_or_post,
                 include_index=False,
                 include_image_weight = False,
                 transform = None,
                 include_edge_mask= False,
                 edge_mask_type = '',
                 use_clahe = False

                 ):
        super().__init__()

        ds_path = os.path.join(file_path,'labels.json')
        with open(ds_path) as f:
            ds = json.load(f)
        self.dataset_metadata = ds
        self.dataset_path = file_path

        self.length = len(ds)
        print('dataset length', self.length)
        self.include_index = include_index
        self.label_mask_cache = {}
        self.include_image_weight = include_image_weight
        self.transform = transform
        self.pre_or_post = pre_or_post
        self.include_edge_mask = include_edge_mask
        self.edge_mask_type = edge_mask_type
        self.use_clahe = use_clahe


    def __getitem__(self, index):
        data_sample = self.dataset_metadata[index][self.pre_or_post]

        sample_name = data_sample['file_name']
        input =self._process_input(sample_name)
        label = self._extract_label(data_sample['annotations'], sample_name)
        # label = label[None, ...] # C x H x W
        if self.include_edge_mask:
            # Edge mask is attached to the
            edge_mask = self._load_edge_mask(sample_name)
            label = np.concatenate([edge_mask, label], axis=-1)

        if self.transform:
            image_path = os.path.join(self.dataset_path, sample_name)
            input, label, _ = self.transform([input, label, image_path])


        ret = {
            'x': input,
            'y': label,
            'img_name':sample_name,
        }

        if self.include_index:
            ret['index'] = index
        if self.include_image_weight:
            # Used for oversampling stats
            if hasattr(data_sample, 'image_weight'):
                ret['image_weight'] = data_sample['image_weight']
            else:
                ret['image_weight'] = label.sum()

        return ret

    def _extract_label(self, annotations_set, sample_name):

        # Load preprocessed mask if exist
        mask_path = os.path.join(self.dataset_path, 'label_mask',sample_name)
        if os.path.exists(mask_path):
            mask = imread_cached(mask_path)[...,[0]].astype(np.float32)
            return mask

        building_polygons = []
        for anno in annotations_set:
            building_polygon_xy = np.array(anno['segmentation'][0], dtype=np.int32).reshape(-1, 2)
            building_polygons.append(building_polygon_xy)

        mask2 = np.zeros((1024, 1024, 1), dtype=np.uint8)
        cv2.fillPoly(mask2, building_polygons, 1)
        mask = mask2.astype(np.float32)
        return mask

    def _process_input(self, image_filename):
        if self.use_clahe:
            img_path = os.path.join(self.dataset_path,'clahe', image_filename)
            img = imread_cached(img_path)
            if img is None:
                # For whatever reason the image can be blank
                img_path = os.path.join(self.dataset_path, image_filename)
                img = imread_cached(img_path)
            return img

        else:
            img_path = os.path.join(self.dataset_path, image_filename)
        img = imread_cached(img_path)
        return img

    def _load_edge_mask(self, sample_name):
        '''
        loading edge mask for edge loss computation
        :return:
        '''
        edge_mask_path = os.path.join(self.dataset_path, self.edge_mask_type, sample_name + '.npz')
        if not os.path.exists(edge_mask_path):
            # empty files have no edges
            edge_mask = np.ones((1024, 1024, 1)) / 20
            return edge_mask.astype(np.float32)

        edge_mask = np.load(edge_mask_path)['arr_0'] # arr_0 is the default name for array, kind of stupid
        edge_mask = edge_mask[..., None, ]
        return edge_mask

    def __len__(self):
        return self.length

class Xview2Detectron2DamageLevelDataset(Xview2Detectron2Dataset):

    def __init__(self, file_path,
                 pre_or_post,
                 background_class = 'new-channel',
                 label_format = 'one_hot',
                 *args, **kwargs
                 ):
        super().__init__(file_path, pre_or_post, *args, **kwargs)
        self.background_class = background_class
        self.label_format = label_format

    def _extract_label(self, annotations_set, sample_name):
        # TODO This data can be preprocessed
        # TODO Resolve overlapping data in preprocessed data
        NUM_CLASSES = 4

        buildings_polygons = [[] for _ in range(NUM_CLASSES)]

        # Distribute buildings to their respective classes
        negative_damage_level_count = 0
        unclassified_damage_level_count = 0
        for anno in annotations_set:
            damage_level = anno['damage_level']

            # assert damage_level >= -1, 'damage level error, did you use positive damage?'

            if damage_level == 4:
                damage_level = 0
                # TODO Decide what to do with unclassified labels

            building_polygon_xy = np.array(anno['segmentation'][0], dtype=np.int32).reshape(-1, 2)
            buildings_polygons[damage_level].append(building_polygon_xy)

        masks = []
        for class_idx, building_poly in enumerate(buildings_polygons):
            mask = np.zeros((1024, 1024,), dtype=np.uint8)
            cv2.fillPoly(mask, building_poly, 1)
            masks.append(mask)
        masks = np.dstack(masks).astype(np.float32)
        if self.background_class == 'new-class':
            positive_px = masks.sum(axis=-1, keepdims=True)
            bg = 1 - positive_px
            masks = np.concatenate([masks, bg], axis = -1)
        elif self.background_class == 'no-damage':
            # all the background pixels grouped with no-damage
            positive_px = masks[..., 1:].sum(axis=-1)
            masks[..., 0] = 1 - positive_px
        else:
            masks = masks

        if self.label_format == 'ordinal':
            for i in range(masks.shape[-1]):
                masks[..., :i] += masks[..., i]
            masks = masks[..., 1:]
            masks.clip(0, 1)

        return masks

