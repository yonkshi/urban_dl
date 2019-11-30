# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TensorMask Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import json

import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.data import DatasetCatalog, MetadataCatalog

from tensormask import add_tensormask_config


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_tensormask_config(cfg)

    # Setting this so default_setup() is happy later on
    args.config_file = f'tensormask/configs/{args.config_file}.yaml'
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.log_dir:
        cfg.OUTPUT_DIR_BASE = args.log_dir
    if args.data_dir:
        cfg.DATASETS.TRAIN = (args.data_dir,)

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print('device:', cfg.MODEL.DEVICE)


    register_datasets(cfg.DATASETS.TRAIN)
    register_datasets(cfg.DATASETS.TEST)

    # setup up logging directory
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR_BASE, args.config_file)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def register_datasets(dsets):
    for dset_name in dsets:
        DatasetCatalog.register(dset_name, lambda: get_building_dicts(dset_name))
        MetadataCatalog.get(dset_name).set(thing_classes=["buildings"])

def get_building_dicts(img_dir, transform=False):
    json_file = os.path.join(img_dir, "labels.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for v in imgs_anns:
        record = {}
        filename = os.path.join(img_dir, v["file_name"])

        # Convert relative file name to absolute filename
        v["file_name"] = filename
        # Convert bbox mode to objects
        v['image_id'] = v['img_id']
        for anno in v['annotations']:
            anno['bbox_mode'] = BoxMode.XYXY_ABS

    print('metadata loading complete!')
    return imgs_anns

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
