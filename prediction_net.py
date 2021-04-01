#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepLab Prediction Script.
"""

import tempfile

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor

from deeplab import add_deeplab_config


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
    cfg = setup(args)

    img = cv2.imread(args.image)

    pred = DefaultPredictor(cfg)
    outputs = pred(img)
    outputs = outputs["sem_seg"].to("cpu")
    output = torch.argmax(outputs, 0).detach().numpy()

    # TODO Transform the output to c array and then create binary image for each class


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

    output_dir.cleanup()
