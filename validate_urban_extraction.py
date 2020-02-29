from pathlib import Path

import gdal
import osr
import tifffile
import rasterio
from rasterio.merge import merge

from unet import UNet
from unet import dataloader
from experiment_manager.config import new_config
from preprocessing import urban_extraction as ue
from torch.utils import data as torch_data
from unet.augmentations import *
from torchvision import transforms

import matplotlib as mpl

# plotting
import matplotlib.pyplot as plt


# reading in geotiff file as numpy array
def read_tif(file: Path):

    if not file.exists():
        raise FileNotFoundError(f'File {file} not found')

    ds = gdal.Open(str(file))

    geotransform = ds.GetGeoTransform()

    proj = osr.SpatialReference(wkt=ds.GetProjection())
    epsg = int(proj.GetAttrValue('AUTHORITY' ,1))

    xy_shape = np.array(ds.GetRasterBand(1).ReadAsArray()).shape

    # get number of bands in raster file
    n_bands = ds.RasterCount

    # initialize a data cube
    xyz_shape = xy_shape + (n_bands,)
    data_cube = np.ndarray(xyz_shape)

    # fill it with bands
    for i in range(1, n_bands+1):
        data_cube[: ,: , i -1] = np.array(ds.GetRasterBand(i).ReadAsArray())

    ds = None
    return data_cube, geotransform, epsg
    # end of read in datacube function


# writing an array to a geo tiff file
def write_tif(arr, geotransform, epsg, save_dir: Path, fname: str, dtype=gdal.GDT_Float32):

    if not save_dir.exists():
        save_dir.mkdir()
    file = save_dir / f'{fname}.tif'

    n_rows, n_cols = arr.shape[:2]
    n_bands = arr.shape[2] if len(arr.shape) > 2 else 1

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(file), n_rows, n_cols, n_bands, dtype)

    # setting coordinate reference system
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())

    # write data to file
    if n_bands == 1:
        arr = arr if len(arr.shape) == 2 else arr[:, :, 0]
        ds.GetRasterBand(1).WriteArray(arr)
    else:
        for i in range(n_bands):
            ds.GetRasterBand(i + 1).WriteArray(arr[:, :, i])

    ds.FlushCache()  # saves to disk
    del driver
    del ds



def load_cfg(configs_dir: Path, experiment: str):
    cfg = new_config()
    cfg.merge_from_file(str(configs_dir / f'{experiment}.yaml'))
    return cfg


def load_net(cfg, models_dir: Path, experiment: str):
    net = UNet(cfg)
    state_dict = torch.load(models_dir / f'{experiment}.pkl', map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)

    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(mode)

    net.to(device)
    net.eval()

    return net

def load_dataset(cfg, data_dir: Path):
    trfm = transforms.Compose([Npy2Torch()])
    dataset = dataloader.UrbanExtractionDataset(cfg, data_dir, transform=trfm)
    return dataset





def classify_tiles(configs_dir: Path, models_dir: Path, root_dir: Path, save_dir: Path, experiment: str):

    classification_batch_size = 10

    cfg = load_cfg(configs_dir, experiment)
    cfg.DATALOADER.SHUFFLE = False
    cfg.TRAINER.BATCH_SIZE = classification_batch_size
    cfg.MODEL.IN_CHANNELS = 3
    cfg.DATALOADER.S2_FEATURES = ['Green_median', 'Red_median', 'NIR_median']
    print(cfg)
    net = load_net(cfg, models_dir, experiment)

    # classification loop
    for train_test in ['train', 'test']:
        dataset = load_dataset(cfg, root_dir / train_test)
        year = dataset.metadata['year']

        for i in range(len(dataset)):
            item = dataset.__getitem__(i)
            img = item['x']

            metadata = dataset.metadata['samples'][i]
            city = metadata['city']
            patch_id = metadata['patch_id']
            row_id, col_id = patch_id.split('-')
            row_id, col_id = int(row_id), int(col_id)
            tif_file = root_dir / train_test / 'guf' / f'GUF_{metadata["city"]}_{metadata["patch_id"]}.tif'
            print(city, patch_id)
            if city == 'Stockholm':
                if 5376 <= row_id <= 6400 and 9728 <= col_id <= 13056:

                    _, geotransform, epsg = read_tif(tif_file)
                    y_pred = net(img.unsqueeze(0))
                    # y_pred = torch.sigmoid(y_pred)

                    y_pred = y_pred.detach().numpy()
                    y_pred = y_pred[0, 0,]

                    fname = f'pred_{metadata["city"]}_{year}_{metadata["patch_id"]}'
                    write_tif(y_pred, geotransform, epsg, save_dir / experiment, fname)



def combine_tiles(data_dir: Path, city: str, year: int, tile_size=256, top_left=(0, 0)):

    bottom_right = top_left

    # getting total height
    height = 0
    while True:
        file = data_dir / f'pred_{city}_{year}_{bottom_right[0]:010d}-{top_left[1]:010d}.tif'
        if file.exists():
            bottom_right = (bottom_right[0] + tile_size, bottom_right[1])

            ds = gdal.Open(str(file))
            geotransform = ds.GetGeoTransform()
            print(geotransform)

            # proj = osr.SpatialReference(wkt=ds.GetProjection())
            # epsg = int(proj.GetAttrValue('AUTHORITY', 1))

        else:
            break
    while True:
        file = data_dir / f'pred_{city}_{year}_{top_left[0]:010d}-{bottom_right[1]:010d}.tif'
        if file.exists():
            bottom_right = (bottom_right[0], bottom_right[1] + tile_size)
        else:
            break

    shape = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
    arr = np.empty(shape, dtype=np.float16)
    # populating arr with classification results
    for i, m in enumerate(range(top_left[0], bottom_right[0], tile_size)):
        for j, n in enumerate(range(top_left[1], bottom_right[1], tile_size)):
            patch_file = data_dir / f'pred_{city}_{year}_{m:010d}-{n:010d}.tif'
            patch_pred = tifffile.imread(str(patch_file))
            arr[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = patch_pred

    classified_map = save_dir / f'pred_{city}_{year}.png'
    mpl.image.imsave(classified_map, arr, vmin=-20, vmax=0, cmap='viridis')


def merge_tiles(data_dir: Path, save_dir: Path, experiment: str, city: str, year: int):

    # getting all files in directory
    files_in_dir = [file for file in data_dir.glob('**/*')]

    # sub setting files
    files_to_mosaic = []
    for file in files_in_dir:
        _, city_file, year_file, _ = file.stem.split('_')
        if city_file == city and year_file == str(year):
            src = rasterio.open(str(file))
            files_to_mosaic.append(src)

    # merging all files to a mosaic
    mosaic, out_trans = merge(files_to_mosaic)

    # getting metadata
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans})

    out_file = save_dir / f'pred_{city}_{year}_{experiment}.tif'
    with rasterio.open(out_file, "w", **out_meta) as dest:
        dest.write(mosaic)



if __name__ == '__main__':

    configs_dir = Path('configs/urban_extraction')
    models_dir = Path('C:/Users/shafner/models')
    root_dir = Path('C:/Users/shafner/projects/urban_extraction/data/preprocessed/urban_extraction_twocities')
    save_dir = Path('C:/Users/shafner/projects/urban_extraction/data/classifications')

    experiment = 's2_GRNIRBands'

    """
    classify_tiles(
        configs_dir=models_dir,
        models_dir=models_dir,
        root_dir=root_dir,
        save_dir=save_dir,
        experiment=experiment,
    )
    """

    city = 'Stockholm'
    year = 2017

    # combine_tiles(save_dir / experiment, 'Stockholm', 2017, top_left=(5376, 9728))
    merge_tiles(save_dir / experiment, save_dir, experiment, city, year)


