import os
import json
import numpy as np

from unet.dataloader import Xview2Detectron2DamageLevelDataset

def main():
    dataset_path = "/storage2/david/datasets/full_new/train/"
    dataset = Xview2Detectron2DamageLevelDataset(dataset_path,
                                                      pre_or_post='post',
                                                      include_image_weight=True,
                                                      background_class='new-class',)

    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            print(f'Step {i}/{len(dataset)}')
        dataset.dataset_metadata[i]['post']['image_weight_per_class'] = list(sample['image_weight_per_class'])

    ds_path = os.path.join(dataset_path, 'labels_new.json')
    with open(ds_path, 'w') as f:
        json.dump(dataset.dataset_metadata, f)


if __name__ == '__main__':
    main()