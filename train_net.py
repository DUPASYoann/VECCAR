#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepLab Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import torch
import cv2
from detectron2.utils.visualizer import Visualizer
import os
import glob
from datetime import datetime
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.utils.analysis import parameter_count_table
from detectron2.utils.events import TensorboardXWriter, get_event_storage
from deeplab import Trainer, add_deeplab_config, build_lr_scheduler
from detectron2.modeling import build_model
import pickle
import numpy as np


def get_cartography_dicts(img_dir):
    """
    Create the dictionary of the dataset
    """

    img_path = os.path.join(img_dir, "Img/*")
    dataset_dicts = []
    for idx, img in enumerate(glob.glob(img_path)):
        image = cv2.imread(img)
        img_height, img_width = image.shape[:2]
        record = {"file_name": img, "image_id": idx, "height": img_height, "width": img_width,
                  "sem_seg_file_name": os.path.join(img_dir, "GT/", os.path.basename(img))}
        dataset_dicts.append(record)
    return dataset_dicts


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
        DatasetCatalog.register("carte_" + d, lambda d=d: get_cartography_dicts("datasets/cartographie/dataset/" + d))
        MetadataCatalog.get("carte_" + d).set(stuff_classes=["background", "foret", "autoroute", "route"],
                                              evaluator_type="sem_seg",
                                              stuff_colors=[(140, 34, 140), (34, 139, 34), (255, 20, 147),
                                                            (218, 165, 32)])

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    # trainer.checkpointer.load("/home/yoann/PycharmProjects/VECCAR/experiment/2021_03_10__20:09:29/model_final.pth")
    trainer.resume_or_load(resume=True)
    trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join("/home/yoann/PycharmProjects/VECCAR/", cfg.OUTPUT_DIR, "model_final.pth")

    # trainer = Trainer(cfg)
    # model = trainer.build_model(cfg)
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load("/home/yoann/PycharmProjects/VECCAR/weights/deeplab_without_last_layer.pkl")
    # checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    # checkpointer.save("model_999")
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # trainer = Trainer(cfg)
    # trainer.resume_or_load(resume=False)
    # model = trainer.build_model(cfg)
    # for param in model.sem_seg_head.decoder.parameters():
    #     param.requires_grad = False
    # torch.nn.init.xavier_uniform_(model.sem_seg_head.predictor.weight)
    # torch.nn.init.uniform_(model.sem_seg_head.predictor.bias)
    # trainer.train()

    # visualisation de la verité terrain
    data = get_cartography_dicts("datasets/cartographie/dataset/val")[0]
    carto_metadata = MetadataCatalog.get("carte_val")
    img = cv2.imread(data["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=carto_metadata, scale=1)
    out = visualizer.draw_dataset_dict(data)
    cv2.imwrite("output/groundtruth.png", out.get_image()[:, :, ::-1])

    # prediction et visualisateur
    pred = DefaultPredictor(cfg)
    outputs = pred(img)
    print(outputs)
    outputs = outputs["sem_seg"].to("cpu")
    print(np.shape(outputs))
    output = torch.argmax(outputs, 0)
    print(output)
    v = Visualizer(img[:, :, ::-1], metadata=carto_metadata, scale=1)
    out = v.draw_sem_seg(output)
    out = out.get_image()[:, :, ::-1]
    cv2.imwrite("./output/result_rgb_annotated_" + datetime.now().strftime("%d_%m_%Y__%H:%M:%S") + "_.jpg", out)
    print(data["file_name"])

    return None


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    # DEBUG ONLY
    args.config_file = "configs/Veccar/Veccar.yaml"
    args.eval_only = False
    args.num_gpus = 1
    args.opts = [ # "MODEL.WEIGHTS", "/home/yoann/PycharmProjects/VECCAR/experiment/2021_03_10__20:09:29/model_final.pth",
                 "OUTPUT_DIR", os.path.join("experiment", datetime.now().strftime("%Y_%m_%d__%H:%M:%S"))]
    # DEBUG ONLY

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
