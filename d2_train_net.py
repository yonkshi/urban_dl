# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import torch
from argparse import ArgumentParser
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode



import os
from os import path

import json

import itertools

import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.utils.logger import setup_logger

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

def register_datasets(dsets):
    for dset_name in dsets:
        DatasetCatalog.register(dset_name, lambda: get_building_dicts(dset_name))
        MetadataCatalog.get(dset_name).set(thing_classes=["buildings"])


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)

    if args.log_dir:
        cfg.OUTPUT_DIR = args.log_dir
    if args.data_dir:
        cfg.DATASETS.TRAIN = (args.data_dir,)

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print('device:', cfg.MODEL.DEVICE)
    cfg.OUTPUT_DIR = path.join(cfg.OUTPUT_DIR, args.config_file)
    cfg.freeze()

    register_datasets(cfg.DATASETS.TRAIN)
    register_datasets(cfg.DATASETS.TEST)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger()
    return cfg


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
        record['image_id'] = v["file_name"]

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

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data-dir', dest='data_dir', type=str,
                      default='/datasets/xview2/small_detectron2_train/', help='dataset directory')
    parser.add_argument('-o', '--log-dir', dest='log_dir', type=str,
                      default='/logs/detectron/', help='logging directory')

    parser.add_argument('-c', '--config', dest='config_file', type=str,
                      default='overfitter_mask_rcnn_r_50_fpn_3x_small_anchor', help='logging directory')

    (options, args) = parser.parse_known_args()
    return options

def main2():

    args = get_args()
    DATASET_LOCATION = args.data_dir
    print('dataset location', DATASET_LOCATION)
    DatasetCatalog.register(DATASET_LOCATION , lambda : get_building_dicts(DATASET_LOCATION))
    MetadataCatalog.get(DATASET_LOCATION).set(thing_classes=["buildings"])


    cfg = get_cfg()
    cfg.merge_from_file(f"configs/{args.config_file}.yaml")
    cfg.DATASETS.TRAIN = (DATASET_LOCATION,)
    cfg.OUTPUT_DIR = os.path.join(args.log_dir, args.config_file)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    # cfg.MODEL.WEIGHTS = os.path.join(args.log_dir, 'model_final.pth')  # initialize from model zoo
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

