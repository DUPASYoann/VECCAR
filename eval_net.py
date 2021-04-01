#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepLab Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import tempfile

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import default_argument_parser, default_setup, launch

from CartoSem import dataset_carto
from deeplab import Trainer, add_deeplab_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg


def main(args):

    for d in ["train", "val"]:
        DatasetCatalog.register("carte_" + d, lambda d=d: dataset_carto.get_cartography_dicts("dataset/" + d))
        MetadataCatalog.get("carte_" + d).set(stuff_classes=["background", "foret", "autoroute", "route"],
                                              evaluator_type="sem_seg",
                                              stuff_colors=[(140, 34, 140), (34, 139, 34), (255, 20, 147),
                                                            (218, 165, 32)])

    cfg = setup(args)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = Trainer.test(cfg, model)
    return res


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    # DEBUG ONLY (DELETE FOR PRODUCTION)
    weights = "/home/yoann/PycharmProjects/VECCAR/experiment/2021_03_31__22:12:49 - 100 fois 1 moins Wi/model_final.pth"
    output_dir = tempfile.TemporaryDirectory()

    args.config_file = "configs/Veccar/Veccar.yaml"
    args.eval_only = False
    args.num_gpus = 1
    args.image = "/home/yoann/PycharmProjects/VECCAR/datasets/cartographie/other/Tours_test.png"
    args.opts = ["MODEL.WEIGHTS", weights,
                 "OUTPUT_DIR", output_dir.name]
    # DEBUG ONLY

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
