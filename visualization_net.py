#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepLab Visualization Script.
"""

import os
import tempfile
from datetime import datetime

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from CartoSem import dataset_carto
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


# noinspection DuplicatedCode
def main(args):
    # Register dataset to catalog with its metadata
    for d in ["train", "val"]:
        DatasetCatalog.register("carte_" + d,
                                lambda d=d: dataset_carto.get_cartography_dicts("dataset" + d))
        MetadataCatalog.get("carte_" + d).set(stuff_classes=["background", "foret", "autoroute", "route"],
                                              evaluator_type="sem_seg",
                                              stuff_colors=[(140, 34, 140), (34, 139, 34), (255, 20, 147),
                                                            (218, 165, 32)])

    cfg = setup(args)

    path = os.path.join("visualisation/", datetime.now().strftime("%d_%m_%Y__%H:%M:%S"))
    os.mkdir(path)

    metadata = MetadataCatalog.get("carte_val")
    pred = DefaultPredictor(cfg)
    for i, data in enumerate(dataset_carto.get_cartography_dicts("dataset/val")):
        print(f"Processing {i} image")

        # GT
        img = cv2.imread(data["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        out = visualizer.draw_dataset_dict(data)
        cv2.imwrite(os.path.join(path, f"Img_{i}_GT.jpg"), out.get_image()[:, :, ::-1])

        # Prediction
        outputs = pred(img)
        outputs = outputs["sem_seg"].to("cpu")
        output = torch.argmax(outputs, 0)
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        out = v.draw_sem_seg(output)
        out = out.get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(path, f"Img_{i}.jpg"), out)


# noinspection DuplicatedCode
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    # DEBUG ONLY
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
