import argparse
import copy
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from lib.config import cfg
from lib.data import build_data_loader
from lib.engine.inference import two_models_inference
from lib.models.model import build_model
from lib.utils.checkpoint import Checkpointer
from lib.utils.directory import makedir
from lib.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Image-Text Matching Inference"
    )
    parser.add_argument(
        "--tgr-config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--vcr-config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--tgr-ckpt-file",
        default="",
        metavar="FILE",
        help="path to checkpoint file",
        type=str,
    )
    parser.add_argument(
        "--vcr-ckpt-file",
        default="",
        metavar="FILE",
        help="path to checkpoint file",
        type=str,
    )
    parser.add_argument(
        "--vcr-dataset",
        default="fashionpedia_outfit_test",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    tgr_cfg = copy.deepcopy(cfg)
    vcr_cfg = copy.deepcopy(cfg)

    tgr_cfg.merge_from_file(args.tgr_config_file)
    tgr_cfg.merge_from_list(args.opts)
    tgr_cfg.freeze()

    vcr_cfg.merge_from_file(args.vcr_config_file)
    vcr_cfg.freeze()

    tgr_model = build_model(tgr_cfg)
    vcr_model = build_model(vcr_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tgr_model.to(device)
    vcr_model.to(device)

    tgr_output_dir = args.tgr_config_file[8:-5]
    vcr_output_dir = args.vcr_config_file[8:-5]

    tgr_checkpointer = Checkpointer(
        tgr_model, save_dir=os.path.join("./output", tgr_output_dir)
    )
    vcr_checkpointer = Checkpointer(
        vcr_model, save_dir=os.path.join("./output", vcr_output_dir)
    )
    _ = tgr_checkpointer.load(args.tgr_ckpt_file)
    _ = vcr_checkpointer.load(args.vcr_ckpt_file)

    dataset_names = tgr_cfg.DATASETS.TEST
    assert len(dataset_names) == 1

    output_folder = os.path.join("./output", tgr_output_dir + vcr_output_dir)
    makedir(output_folder)

    tgr_data_loader = build_data_loader(tgr_cfg, is_train=False, is_distributed=None)[0]
    vcr_data_loader = build_data_loader(vcr_cfg, is_train=False, is_distributed=None)[0]

    logger = setup_logger("CompFashion", output_folder, 0)
    logger.info("Start two models inference evaluation.")

    two_models_inference(
        tgr_model,
        vcr_model,
        tgr_data_loader,
        vcr_data_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
