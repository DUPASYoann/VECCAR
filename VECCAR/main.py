from detectron2.engine import default_argument_parser

if __name__ == "__main__" :
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    args.config_file = "configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml"