#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import torch
import json
import tifffile
from pathlib import Path
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
        edge_mask_path = os.path.join(self.dataset_path, 'edge_loss_weight_mask', sample_name + '.npz')
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
                 *args, **kwargs
                 ):
        super().__init__(file_path, pre_or_post, *args, **kwargs)
        self.background_class = background_class

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

        return masks



class UrbanExtractionDataset(torch.utils.data.Dataset):
    '''
    Dataset for Urban Extraction style labelled Dataset
    '''
    def __init__(self, cfg, # used for feature names
                 root_dir: Path,  # path to folder with sub folder images and labels and a metadata file
                 include_index: bool = False,  # index of sample
                 transform: list = None  # list of transformations
                 ):
        super().__init__()

        # setting up directories
        self.root_dir = Path(root_dir)
        self.s1_dir = self.root_dir / 'sentinel1'
        self.s2_dir = self.root_dir / 'sentinel2'
        self.label_dir = self.root_dir / cfg.DATALOADER.LABEL
        self.cfg = cfg

        # loading metadata of dataset
        with open(str(self.root_dir / 'metadata.json')) as f:
            metadata = json.load(f)
        self.metadata = metadata
        self.year = metadata['year']

        self.length = len(self.metadata['samples'])
        print('dataset length', self.length)

        self.include_index = include_index
        self.transform = transform

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_feature_selection = self._get_feature_selection(metadata['sentinel1'],
                                                                cfg.DATALOADER.S1_FEATURES)
        self.s2_feature_selection = self._get_feature_selection(metadata['sentinel2'],
                                                                cfg.DATALOADER.S2_FEATURES)

    def __getitem__(self, index):

        # loading metadata of sample
        sample_metadata = self.metadata['samples'][index]

        city = sample_metadata['city']
        patch_id = sample_metadata['patch_id']

        s1_file = self.s1_dir / f'S1_{city}_{self.year}_{patch_id}.tif'
        s2_file = self.s2_dir / f'S2_{city}_{self.year}_{patch_id}.tif'
        label_file = self.label_dir / f'{str(self.cfg.DATALOADER.LABEL).upper()}_{city}_{patch_id}.tif'

        # loading images and corresponding label
        if not any(self.s1_feature_selection):  # only sentinel 2 features
            # img = cv2.imread(str(s1_file), -1)
            img = tifffile.imread(str(s2_file))
            img = img[:, :, self.s2_feature_selection]
        elif not any(self.s2_feature_selection):  # only sentinel 1 features
            # img = cv2.imread(str(s1_file), -1)
            img = tifffile.imread(str(s1_file))
            img = img[:, :, self.s1_feature_selection]
        else:  # sentinel 1 and sentinel 2 features
            # s1_img = cv2.imread(str(s1_file), -1)
            s1_img = tifffile.imread(str(s1_file))
            s1_img = s1_img[:, :, self.s1_feature_selection]
            # s2_img = cv2.imread(str(s2_file), -1)
            s2_img = tifffile.imread(str(s2_file))
            s2_img = s2_img[:, :, self.s2_feature_selection]
            img = np.stack([s1_img, s2_img], axis=-1)

        label = tifffile.imread(str(label_file))
        label = label[:, :, None]
        # label_old = cv2.imread(str(label_file), 0)

        if self.transform:
            img,label,sample_id, = self.transform((img, label, patch_id,))

        sample = {
            'x': img.float(), # numpy.array (m, n, N_CHANNELS)
            'y': label.float(), # numpy.array (m, n, 1)
            'img_name': sample_id, # identifier of sample
            'image_weight': np.float(sample_metadata['img_weight'])
        }

        if self.include_index:
            sample['index'] = index

        return sample


    def _get_feature_selection(self, features, selection):
        feature_selection = [False for _ in range(len(features))]
        for feature in selection:
            i = features.index(feature)
            feature_selection[i] = True
        return feature_selection

    def __len__(self):
        return self.length
