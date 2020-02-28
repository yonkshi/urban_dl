from pathlib import Path

import gdal
import osr

from unet import UNet
from unet import dataloader
from experiment_manager.config import new_config
from preprocessing import urban_extraction as ue
from torch.utils import data as torch_data
from unet.augmentations import *
from torchvision import transforms

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


# function to write a data cube to a geo tiff file
def write_tif(data_cube, geotransform, epsg, save_dir: Path, fname):

    file = save_dir / f'{fname}.tif'

    n_rows, n_cols = data_cube.shape[0:2]
    n_bands = data_cube.shape[2] if len(data_cube.shape ) >2 else 1

    # open geo tiff file
    ds = gdal.GetDriverByName('GTiff').Create(str(file), n_cols, n_rows, n_bands, )
    ds.SetGeoTransform(geotransform)

    # set crs
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())

    # write data cube to geo tiff
    if n_bands == 1:
        ds.GetRasterBand(1).WriteArray(data_cube[: ,: ,0])
    else:
        for i_band in range(n_bands):
            ds.GetRasterBand(i_band +1).WriteArray(data_cube[: ,: ,i_band])

    dst_ds = gdal.GetDriverByName('GTiff').CreateCopy(str(file), ds)
    dst_ds = None


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
    for train_test in ['test']:
        dataset = load_dataset(cfg, root_dir / train_test)

        """
        dataloader_kwargs = {
            'batch_size': 8,
            'num_workers': 8,
            'shuffle': False,
            'drop_last': False,
            'pin_memory': True,
        }

        dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
        

        # classification loop
        for i, batch in enumerate(dataloader):
            imgs = batch['x']

            print(imgs.shape)
        """

        for i in range(len(dataset)):
            item = dataset.__getitem__(i)
            img = item['x']

            metadata = dataset.metadata['samples'][i]
            city = metadata['city']
            patch_id = metadata['patch_id']
            file_id = f'{metadata["city"]}'
            tif_file = root_dir / train_test / 'guf' / f'GUF_{metadata["city"]}_{metadata["patch_id"]}.tif'

            _, geotransform, epsg = read_tif(tif_file)
            y_pred = net(img.unsqueeze(0))
            # y_pred = torch.sigmoid(y_pred)

            y_pred = y_pred.detach().numpy()
            y_pred = y_pred[0, 0,]

            write_tif(y_pred, geotransform, epsg, save_dir / experiment, fname):



def combine_tiles(data_dir: Path, city: str, year: int):
    pass

def classify_tiles_depecated(net, dataloader):

    cfg_file = configs_dir / f'{experiment}.yaml'
    model_file = models_dir / f'{experiment}.pkl'

    # loading config
    cfg = new_config()
    cfg.merge_from_file(str(cfg_file))
    print(cfg)

    # loading data
    trfm = []
    trfm.append(Npy2Torch())
    trfm = transforms.Compose(trfm)

    dataset = dataloader.UrbanExtractionDataset(cfg, data_dir, transform=trfm)
    sample = dataset.__getitem__(index)
    img = sample['x'][None,]
    label = sample['y']

    # loading network
    net = UNet(cfg)
    mode = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    device = torch.device(mode)

    net.to(device)
    net.eval()


    y_pred = net(img)
    print(y_pred.shape)

    return np.empty((1,1))


if __name__ == '__main__':

    configs_dir = Path('configs/urban_extraction')
    models_dir = Path('C:/Users/shafner/models')
    root_dir = Path('C:/Users/shafner/projects/urban_extraction/data/preprocessed/urban_extraction_twocities')
    save_dir = Path('C:/Users/shafner/projects/urban_extraction/data/classifications')

    experiment = 's2_GRNIRBands'

    classify_tiles(
        configs_dir=models_dir,
        models_dir=models_dir,
        root_dir=root_dir,
        save_dir=save_dir,
        experiment=experiment,
    )



