from pathlib import Path

import tifffile

from unet import UNet
from unet import dataloader
from experiment_manager.config import new_config
from preprocessing import urban_extraction as ue
from torch.utils import data as torch_data
from unet.augmentations import *
from torchvision import transforms

# plotting
import matplotlib.pyplot as plt


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

def load_dataloader(cfg, data_dir: Path):
    trfm = transforms.Compose([Npy2Torch()])
    dataset = dataloader.UrbanExtractionDataset(cfg, data_dir, transform=trfm)
    return dataset





def classify_tiles(configs_dir: Path, models_dir: Path, root_dir: Path, save_dir: Path, experiment: str):

    classification_batch_size = 10

    cfg = load_cfg(configs_dir, experiment)
    cfg.DATALOADER.SHUFFLE = False
    cfg.TRAINER.BATCH_SIZE = classification_batch_size

    net = load_net(cfg, models_dir, experiment)

    # classification loop
    for dataset in ['test']:
        dataloader = load_dataloader(cfg, root_dir / dataset)
        # classification loop
        for i in range(len(dataloader)):
            item = dataloader.__getitem__(i)
            img = item['x']

            metadata = dataloader.metadata['samples'][i]
            city = metadata['city']
            patch_id = metadata['patch_id']
            tif_file = root_dir / dataset / f'GUF_{metadata["city"]}_{metadata["patch_id"]}.tif'



            y_pred = net(img)



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
        configs_dir=configs_dir,
        models_dir=models_dir,
        root_dir=root_dir,
        save_dir=save_dir,
        experiment=experiment,
    )



