import shutil, json, cv2
from pathlib import Path
import numpy as np
import tifffile

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
    arr = tifffile.imread(str(file))
    arr = np.array(arr)
    if arr.shape[0] == tile_size and arr.shape[1] == tile_size:
        return False
    return True


def preprocess_dataset(root_dir: Path, save_dir: Path, experiment_name: str, year: int, cities: list,
                       s1_features: list, s2_features: list, split: float):

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
                new_file = train_test_dir / product / file.name
                if not new_file.parent.exists():
                    new_file.parent.mkdir(parents=True)
                shutil.copy(str(file), str(train_test_dir / product / file.name))

    # writing metadata to .json file for train and test set
    dataset_metadata['dataset'] = 'train'
    dataset_metadata['samples'] = train_samples
    with open(str(train_dir / 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_metadata, f, ensure_ascii=False, indent=4)

    dataset_metadata['dataset'] = 'test'
    dataset_metadata['samples'] = test_samples
    with open(str(test_dir / 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_metadata, f, ensure_ascii=False, indent=4)




def write_metadata_file(root_dir: Path, save_dir: Path, year: int, cities: list, s1_features: list, s2_features: list):

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
    with open(str(save_dir / 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_metadata, f, ensure_ascii=False, indent=4)





if __name__ == '__main__':

    # root_dir = Path('C:/Users/shafner/projects/urban_extraction/data/gee/urban_extraction_gee_download')
    # save_dir = Path('C:/Users/shafner/projects/urban_extraction/data/preprocessed/')
    # root_dir = Path('/Midgard/Data/pshi/datasets/sentinel/raw/')
    # save_dir = Path('/Midgard/Data/pshi/datasets/sentinel/preprocessed/')
    # experiment = 'urban_extraction_morecities'

    root_dir = Path('C:/Users/shafner/projects/urban_extraction/data/gee/')
    save_dir = Path('C:/Users/shafner/projects/urban_extraction/data/preprocessed/')
    # root_dir = Path('/Midgard/Data/pshi/datasets/sentinel/raw/')
    # save_dir = Path('/Midgard/Data/pshi/datasets/sentinel/preprocessed/')
    # experiment = 'urban_extraction_twocities'

    metadata_dir = Path('C:/Users/shafner/projects/urban_extraction/data/gee/urban_extraction_2019')


    year = 2019
    cities = ['Stockholm', 'Beijing', 'Milan']

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

    write_metadata_file(
        root_dir=metadata_dir,
        save_dir=metadata_dir,
        year=year,
        cities=cities,
        s1_features=sentinel1_features,
        s2_features=sentinel2_features
    )


    # preprocess_dataset(
    #     root_dir=root_dir,
    #     save_dir=save_dir,
    #     experiment_name=experiment,
    #     year=year,
    #     cities=cities,
    #     s1_features=sentinel1_features,
    #     s2_features=sentinel2_features,
    #     split=split
    # )
