import shutil, json, cv2
from pathlib import Path
from preprocessing.utils import *
import numpy as np


# getting list of feature names based on input parameters
def sentinel1_feature_names(polarizations: list, metrics: list):
    names = []
    for orbit in ['asc', 'desc']:
        for pol in polarizations:
            for metric in metrics:
                names.append(f'{pol}_{orbit}_{metric}')
    return names


# getting list of feature names based on input parameters
def sentinel2_feature_names(bands: list, indices: list, metrics: list):
    band_names = [f'{band}_{metric}' for band in bands for metric in metrics]
    index_names = [f'{index}_{metric}' for index in indices for metric in metrics]
    return band_names + index_names


# computing the percentage of urban pixels for a file
def get_image_weight(file: Path):
    if not file.exists():
        raise FileNotFoundError(f'Cannot find file {file.name}')
    arr = cv2.imread(str(file), 0)
    n_urban = np.sum(arr)
    return int(n_urban)


def is_edge_tile(file: Path, tile_size=256):
    arr, _, _ = read_tif(file)
    arr = np.array(arr)
    if arr.shape[0] == tile_size and arr.shape[1] == tile_size:
        return False
    return True


# preprocessing dataset
def preprocess_dataset(data_dir: Path, save_dir: Path, cities: list, year: int, label: str,
                       s1_features: list, s2_features: list, split: float, seed: int = 42):

    # setting up raw data directories
    s1_dir = data_dir / 'sentinel1'
    s2_dir = data_dir / 'sentinel2'
    label_dir = data_dir / label

    # setting up save dir
    if not save_dir.exists():
        save_dir.mkdir()

    # container to store all the metadata
    dataset_metadata = {
        'cities': cities,
        'year': year,
        'sentinel1': s1_features,
        'sentinel2': s2_features,
    }

    # getting all guf files
    label_files = [file for file in label_dir.glob('**/*')]

    # generating random numbers for split
    np.random.seed(seed)
    random_numbers = list(np.random.uniform(size=len(label_files)))

    # main loop splitting into train test, removing edge tiles, and collecting metadata
    samples = {'train': [], 'test': []}
    for i, (label_file, random_num) in enumerate(zip(label_files, random_numbers)):
        if not is_edge_tile(label_file):

            sample_metadata = {}

            _, city, patch_id = label_file.stem.split('_')

            sample_metadata['city'] = city
            sample_metadata['patch_id'] = patch_id
            sample_metadata['img_weight'] = get_image_weight(label_file)

            s1_file = s1_dir / f'S1_{city}_{year}_{patch_id}.tif'
            s2_file = s2_dir / f'S2_{city}_{year}_{patch_id}.tif'

            if random_num > split:
                train_test_dir = save_dir / 'train'
                samples['train'].append(sample_metadata)
            else:
                train_test_dir = save_dir / 'test'
                samples['test'].append(sample_metadata)

            if not train_test_dir.exists():
                train_test_dir.mkdir()

            # copying all files into new directory
            for file, product in zip([label_file, s1_file, s2_file], [label, 'sentinel1', 'sentinel2']):
                new_file = train_test_dir / product / file.name
                if not new_file.parent.exists():
                    new_file.parent.mkdir()
                shutil.copy(str(file), str(train_test_dir / product / file.name))

    # writing metadata to .json file for train and test set
    for train_test in ['train', 'test']:
        dataset_metadata['dataset'] = train_test
        dataset_metadata['samples'] = samples[train_test]
        metadata_file = save_dir / train_test / 'metadata.json'
        with open(str(metadata_file), 'w', encoding='utf-8') as f:
            json.dump(dataset_metadata, f, ensure_ascii=False, indent=4)



def write_metadata_file(root_dir: Path, year: int, cities: list, s1_features: list, s2_features: list):

    # TODO: use this function also in the main preprocessing function to generate metadata file

    # setting up raw data directories
    s1_dir = root_dir / 'sentinel1'

    # container to store all the metadata
    dataset_metadata = {
        'cities': cities,
        'year': year,
        'sentinel1': s1_features,
        'sentinel2': s2_features,
    }

    # getting all sentinel1 files
    s1_files = [file for file in s1_dir.glob('**/*')]

    # main loop splitting into train test, removing edge tiles, and collecting metadata
    samples = []
    for i, s1_file in enumerate(s1_files):
        if not is_edge_tile(s1_file):

            sample_metadata = {}

            _, city, _, patch_id = s1_file.stem.split('_')

            sample_metadata['city'] = city
            sample_metadata['patch_id'] = patch_id

            samples.append(sample_metadata)

    # writing metadata to .json file for train and test set
    dataset_metadata['samples'] = samples
    with open(str(root_dir / 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_metadata, f, ensure_ascii=False, indent=4)





if __name__ == '__main__':

    gee_dir = Path('C:/Users/shafner/projects/urban_extraction/data/gee/')
    save_dir = Path('C:/Users/shafner/projects/urban_extraction/data/preprocessed/')

    cities = ['Beijing', 'Stockholm']
    year = 2017
    label = 'wsf'
    bucket = 'urban_extraction_twocities_raw'
    data_dir = gee_dir / bucket
    save_dir = save_dir / 'urban_extraction_twocities'


    split = 0.2

    # sentinel 1 parameters
    s1params = {
        'polarizations': ['VV', 'VH'],
        'metrics': ['mean']
    }

    # sentinel 2 parameters
    s2params = {
        'bands': ['Blue', 'Green', 'Red', 'RedEdge1', 'RedEdge2', 'RedEdge3', 'NIR', 'RedEdge4', 'SWIR1', 'SWIR2'],
        'indices': [],
        'metrics': ['median']
    }

    # generating feature names for sentinel 1 and sentinel 2
    sentinel1_features = sentinel1_feature_names(polarizations=s1params['polarizations'],
                                                 metrics=s1params['metrics'])
    sentinel2_features = sentinel2_feature_names(bands=s2params['bands'],
                                                 indices=s2params['indices'],
                                                 metrics=s2params['metrics'])

    preprocess_dataset(data_dir, save_dir, cities, year, label, sentinel1_features, sentinel2_features, split)

    # cities = ['Stockholm', 'Beijing', 'Milan']
    # year = 2019
    # root_dir = Path('/storage/shafner/urban_extraction/urban_extraction_2019')
    # write_metadata_file(
    #     root_dir=root_dir,
    #     year=year,
    #     cities=cities,
    #     s1_features=sentinel1_features,
    #     s2_features=sentinel2_features
    # )



