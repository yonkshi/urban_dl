import shutil, json, cv2
from pathlib import Path
import numpy as np
import tifffile
from matplotlib import image


def tif_to_png(tif_file: Path, indices: list, save_dir: Path, fname: str):

    tif_img = tifffile.imread(str(tif_file))
    tif_shape = tif_img.shape
    if len(tif_shape) == 2:
        tif_img = tif_img[:, :, None]
    rgb_img = np.empty((tif_img.shape[0], tif_img.shape[1], 3))

    for rgb_index, tif_index in enumerate(indices):
        rgb_img[:, :, rgb_index] = tif_img[:, :, indices[tif_index]]

    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    png_file = save_dir / f'{fname}.png'
    image.imsave(png_file, rgb_img)


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
    arr = cv2.imread(str(file), 0)
    arr = np.array(arr)
    if arr.shape == (tile_size, tile_size):
        return False
    return True


def preprocess_dataset_png(root_dir: Path, save_dir: Path, experiment_name: str, year: int, cities: list,
                       s1_features: list, s2_features: list, split: float):
    # TODO: preprocessing that saves files as png instead of geotiffs
    # setting up raw data directories
    s1_dir = root_dir / 'sentinel1'
    s2_dir = root_dir / 'sentinel2'
    guf_dir = root_dir / 'guf'

    # setting up directories for preprocessed data
    train_dir = save_dir / experiment_name / 'train'
    test_dir = save_dir / experiment_name / 'test'

    # container to store all the metadata
    dataset_metadata = {
        'cities': cities,
        'year': year,
        'sentinel1': s1_features,
        'sentinel2': s2_features,
    }

    # getting all guf files
    guf_files = [file for file in guf_dir.glob('**/*')]

    # generating random numbers for split
    random_numbers = list(np.random.uniform(size=len(guf_files)))

    # main loop splitting into train test, removing edge tiles, and collecting metadata
    train_samples, test_samples = [], []
    for i, (guf_file, random_num) in enumerate(zip(guf_files, random_numbers)):
        if not is_edge_tile(guf_file):

            sample_metadata = {}

            _, city, patch_id = guf_file.stem.split('_')

            sample_metadata['city'] = city
            sample_metadata['patch_id'] = patch_id
            sample_metadata['img_weight'] = get_image_weight(guf_file)

            s1_file = s1_dir / f'S1_{city}_{year}_{patch_id}.tif'
            s2_file = s2_dir / f'S2_{city}_{year}_{patch_id}.tif'

            if random_num > split:
                train_test_dir = train_dir
                train_samples.append(sample_metadata)
            else:
                train_test_dir = test_dir
                test_samples.append(sample_metadata)

            # copying all files into new directory
            for file, product in zip([guf_file, s1_file, s2_file], ['guf', 'sentinel1', 'sentinel2']):
                save_dir = train_test_dir / product
                if not save_dir.exists():
                    save_dir.mkdir(parents=True)
                if product == 'guf':
                    tif_to_png(file, [0, 0, 0], save_dir, file.fname)
                if product == 'sentinel1':
                    tif_to_png(file, [0, 1, 2], save_dir, file.fname)
                if product == 'sentinel2':
                    tif_to_png(file, [6, 2, 1], save_dir, file.fname)

    # writing metadata to .json file for train and test set
    dataset_metadata['dataset'] = 'train'
    dataset_metadata['samples'] = train_samples
    with open(str(train_dir / 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_metadata, f, ensure_ascii=False, indent=4)

    dataset_metadata['dataset'] = 'test'
    dataset_metadata['samples'] = test_samples
    with open(str(test_dir / 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_metadata, f, ensure_ascii=False, indent=4)
    pass


if __name__ == '__main__':

    root_dir = Path('C:/Users/shafner/projects/urban_extraction/data/gee/')
    save_dir = Path('C:/Users/shafner/projects/urban_extraction/data/preprocessed/')

    experiment = 'urban_extraction_debug'
    year = 2017
    cities = ['Beijing']

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

    preprocess_dataset_png(
        root_dir=root_dir / experiment,
        save_dir=save_dir,
        experiment_name=experiment,
        year=year,
        cities=cities,
        s1_features=sentinel1_features,
        s2_features=sentinel2_features,
        split=split
    )
