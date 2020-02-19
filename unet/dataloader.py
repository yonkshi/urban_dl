#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import torch
import json
from pathlib import Path
from imageio import imread
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
    def __init__(self, root_dir: Path, # path to folder with subfolder images and labels and a metadata file
                 product_name: str,
                 include_index: bool = False, # index of sample
                 transform: list = None # list of transformations
                 ):
        super().__init__()

        self.root_dir = root_dir
        self.product_name = product_name

        # directories for images and labels
        self.images_dir = root_dir / 'images/'
        self.labels_dir = root_dir / 'labels/'

        # loading metadata and subsetting it to cities, year and product
        metadata_file = root_dir / 'metadata.json'
        self.dataset_metadata = self._load_metadata(metadata_file)

        self.length = len(self.dataset_metadata)
        print('dataset length', self.length)

        self.include_index = include_index
        self.transform = transform

    def __getitem__(self, index):

        # loading metadata of sample
        metadata_sample = self.dataset_metadata[index]

        # loading image and corresponding label
        sample_id = metadata_sample['sample_id']
        # TODO: X and y are probably not numpy arrays -> convert to required type
        image = self._load_file(sample_id, self.product_name)
        label = self._load_file(sample_id, 'label')

        # crop image to (1024, 1024)

        sample = {
            'image': image, # numpy.array (m, n, 3)
            'label': label, # numpy.array (m, n, 1)
            'id': sample_id, # identifier of sample
            'urban_percentage': metadata_sample['urban_percentage'] # just copying from metadata
        }

        if self.include_index:
            sample['index'] = index

        if self.transform:
            sample = self.transform(sample)

        return sample

    # getter function for image or label files
    def _load_file(self, file_id: str, product_name: str):

        # construct file name and check for its existence
        file_dir = self.labels_dir if product_name == 'label' else self.images_dir
        file = file_dir / f'{file_id}_{product_name}.png'
        if not file.exists():
            raise FileNotFoundError(f'Cannot find file {file}')

        # loading file
        file_data = np.array(imread(file))
        # TODO: find better solution for file size
        file_data = file_data[:1024,:1024,:]
        if product_name == 'label':
            file_data = file_data[:,:,0] / 255
            file_data = file_data.astype(int)
            return file_data[:,:,None]
        else:
            file_data = file_data[:,:,:3] / 255
            return file_data

    # helper function to load metadata from .json file
    def _load_metadata(self, file:Path):
        # TODO: check if all data is available
        with open(file) as f:
            metadata = json.load(f)
        # TODO: change metadata format to meet requirements
        print(metadata)
        return metadata

    def __len__(self):
        return self.length


if __name__ == '__main__':

    data_dir = Path('C:/Users/shafner/projects/urban_extraction/data/preprocessed/test_dataset/')
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'

    dataset = UrbanExtractionDataset(train_dir, 'S2FC', True)

    index = 100

    sample = dataset.__getitem__(index)
    image = sample['image']
    label = sample['label']
    print(f'Image {index} ({type(image)})')
    print(f'Shape: {image.shape}, Range: [{np.amin(image)}, {np.amax(image)}], Type: {image.dtype}')
    print(f'Label {index} ({type(label)})')
    print(f'Shape: {label.shape}, Range: [{np.amin(label)}, {np.amax(label)}] Type: {label.dtype}')
