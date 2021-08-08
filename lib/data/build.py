import torch.utils.data as tud
import torchvision.transforms as T

from lib.utils.comm import get_world_size

from . import datasets as D
from .collate_batch import collate_fn, init_collate_fn
from .datasets import DatasetCatalog


def build_transform(name, is_train):
    normalizer = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if "fashioniq" or "fashionpedia" in name:
        if is_train:
            transform = T.Compose(
                [
                    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.3)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    # T.Lambda(
                    #     lambda xx: xx + 0.01 * torch.randn(xx.shape)
                    # ),
                    normalizer,
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    normalizer,
                ]
            )
    else:
        NotImplementedError
    return transform


def build_dataset(dataset_list, dataset_catalog, is_train=True, vocab="glove"):
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        args["transform"] = build_transform(dataset_name, is_train)
        args["vocab"] = vocab

        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    if len(datasets) > 1:
        dataset = tud.ConcatDataset(datasets)

    return [dataset]


def build_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return tud.distributed.DistributedSampler(dataset)
    if shuffle:
        sampler = tud.sampler.RandomSampler(dataset)
    else:
        sampler = tud.sampler.SequentialSampler(dataset)
    return sampler


def build_batch_data_sampler(sampler, images_per_batch, is_train=True):
    batch_sampler = tud.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=is_train
    )
    return batch_sampler


def build_data_loader(cfg, is_train=True, is_distributed=False):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus
        )
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus
        )
        images_per_gpu = images_per_batch // num_gpus
        shuffle = is_distributed

    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list, DatasetCatalog, is_train, cfg.MODEL.VOCAB)

    data_loaders = []
    for dataset in datasets:
        sampler = build_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = build_batch_data_sampler(sampler, images_per_gpu, is_train)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = tud.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=init_collate_fn if cfg.MODEL.VOCAB == "init" else collate_fn,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
