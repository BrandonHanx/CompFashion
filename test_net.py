import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from lib.config import cfg
from lib.data import build_data_loader
from lib.engine.inference import inference
from lib.models.model import build_model
from lib.utils.checkpoint import Checkpointer
from lib.utils.comm import get_rank, synchronize
from lib.utils.directory import makedir
from lib.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Image-Text Matching Inference"
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--checkpoint-file",
        default="",
        metavar="FILE",
        help="path to checkpoint file",
        type=str,
    )
    parser.add_argument(
        "--local-rank",
        default=0,
        type=int,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--save-data",
        help="Save predicted results",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--rerank",
        help="Use k-reciprocal post-processing",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_model(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    output_dir = os.path.join("./output", args.config_file[8:-5])
    checkpointer = Checkpointer(model, save_dir=output_dir)
    _ = checkpointer.load(args.checkpoint_file)

    output_folders = list()
    dataset_names = cfg.DATASETS.TEST
    for dataset_name in dataset_names:
        output_folder = os.path.join(output_dir, "inference", dataset_name)
        makedir(output_folder)
        output_folders.append(output_folder)

    logger = setup_logger(
        "CompFashion", os.path.join(output_dir, "inference"), get_rank()
    )
    data_loaders_val = build_data_loader(
        cfg, is_train=False, is_distributed=distributed
    )
    for output_folder, dataset_name, data_loader_val in zip(
        output_folders, dataset_names, data_loaders_val
    ):
        logger.info("Using {} GPUs".format(num_gpus))

        inference(
            model,
            data_loader_val,
            device=device,
            output_folder=output_folder,
            save_data=args.save_data,
            rerank=args.rerank,
        )
        synchronize()


if __name__ == "__main__":
    main()
