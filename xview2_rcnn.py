


# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import torch
from argparse import ArgumentParser

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

setup_logger()

import os
import numpy as np
import json

import itertools


# write a function that loads the dataset into detectron2's standard format
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for _, v in imgs_anns.items():
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def get_building_dicts(img_dir):
    json_file = os.path.join(img_dir, "labels.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for v in imgs_anns:
        record = {}
        filename = os.path.join(img_dir, v["file_name"])
        record["file_name"] = filename
        record["height"] = v['height']
        record["width"] = v['width']

        annos = v["annotations"]
        objs = []
        for polygon_pairs in annos:

            poly = list(itertools.chain.from_iterable(polygon_pairs))
            px, py = np.array(polygon_pairs).T
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    print('metadata loading complete!')
    return dataset_dicts
def get_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data-dir', dest='data_dir', type=str,
                      default='/datasets/xview2/small_detectron2_train/', help='dataset directory')
    parser.add_argument('-o', '--log-dir', dest='log_dir', type=str,
                      default='/logs/detectron/overfitter_mask_rcnn_resnext_101_fpn_1x/', help='logging directory')

    (options, args) = parser.parse_known_args()
    return options

def main():

    args = get_args()
    DATASET_LOCATION = args.data_dir
    print('dataset location', DATASET_LOCATION)
    DatasetCatalog.register(DATASET_LOCATION , lambda : get_building_dicts(DATASET_LOCATION))
    MetadataCatalog.get(DATASET_LOCATION).set(thing_classes=["buildings"])


    cfg = get_cfg()
    cfg.merge_from_file("configs/overfitter_mask_rcnn_resnext_101_fpn_1x.yaml")
    cfg.DATASETS.TRAIN = (DATASET_LOCATION,)
    cfg.OUTPUT_DIR = args.log_dir
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    # cfg.MODEL.WEIGHTS = os.path.join(args.log_dir, 'model_final.pth')  # initialize from model zoo
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()



if __name__ == '__main__':
    main()
