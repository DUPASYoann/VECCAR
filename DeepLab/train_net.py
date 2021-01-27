#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepLab Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch

# Perso
import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot
import matplotlib.image as mpimg
import os
import glob

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler


def get_cartography_dicts(img_dir):
    img_path = os.path.join(img_dir, "Img/*")
    img_height = 512
    img_width = 1024
    dataset_dicts = []
    for idx, img in enumerate(glob.glob(img_path)):
        record = {}
        record["file_name"] = img
        record["image_id"] = idx
        record["height"] = img_height
        record["width"] = img_width
        record["sem_seg_file_name"] = os.path.join(img_dir, "Gt/", os.path.basename(img))
        dataset_dicts.append(record)
    return dataset_dicts


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "sem_seg":
            return SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    # VECCAR
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 4
    cfg.DATASETS.TRAIN = "my_dataset"
    cfg.DATASETS.TEST = ""
    cfg.MODEL.SEM_SEG_HEAD.NORM = "BN"
    # VECCAR

    default_setup(cfg, args)
    return cfg


def main(args):
    DatasetCatalog.register("my_dataset", lambda d=None: get_cartography_dicts("./datasets/cartographie/classic/"))
    MetadataCatalog.get("my_dataset").set(stuff_classes=["background", "foret", "autoroute", "route"],
                                          evaluator_type="cityscapes_sem_seg")
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    im = cv2.imread("datasets/cartographie/other/Tours_test.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pred = DefaultPredictor(cfg)
    outputs = pred(im)
    outputs = outputs["sem_seg"].to("cpu")
    output = torch.argmax(outputs, 0)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN), scale=1.2)
    out = v.draw_sem_seg(output)
    out = out.get_image()[:, :, ::-1]
    cv2.imwrite("./output/result_rgb_annotated_1_.jpg", out)

    return None


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(MetadataCatalog.list())
    print("Command Line Args:", args)
    args.config_file = "configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"
    args.eval_only = False
    args.num_gpus = 1

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
