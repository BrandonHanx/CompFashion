import datetime
import logging
import os
import time
from collections import defaultdict

import torch
from tqdm import tqdm

from lib.data.metrics import evaluation_common, evaluation_cross
from lib.utils.comm import all_gather, is_main_process, synchronize


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = defaultdict(list)
    for batch_data in tqdm(data_loader):
        for k, v in batch_data.items():
            if not k == "meta_info":
                batch_data[k] = v.to(device)

        with torch.no_grad():
            comp_feature, target_feature = model(batch_data)
        for result in output:
            for img_id, pred in zip(image_ids, result):
                results_dict[img_id].append(pred)
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("CompFashion.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
    return predictions


def inference(
    model,
    data_loader,
    dataset_name="cuhkpedes-test",
    device="cuda",
    output_folder="",
    save_data=True,
    rerank=True,
):
    logger = logging.getLogger("CompFashion.inference")
    dataset = data_loader.dataset
    logger.info(
        "Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset))
    )

    predictions = None
    if not os.path.exists(os.path.join(output_folder, "inference_data.npz")):
        # convert to a torch.device for efficiency
        device = torch.device(device)
        num_devices = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        start_time = time.time()

        predictions = compute_on_dataset(model, data_loader, device)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        logger.info(
            "Total inference time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)

        if not is_main_process():
            return

    if not hasattr(model.embed_model, "inference_mode"):
        return evaluation_common(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            save_data=save_data,
            rerank=rerank,
            topk=[1, 5, 10],
        )

    if model.embed_model.inference_mode == "common":
        return evaluation_common(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            save_data=save_data,
            rerank=rerank,
            topk=[1, 5, 10],
        )

    if model.embed_model.inference_mode == "cross":
        assert hasattr(model.embed_model, "get_similarity")
        sim_calculator = model.embed_model.get_similarity
        return evaluation_cross(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            save_data=save_data,
            sim_calculator=sim_calculator,
            topk=[1, 5, 10],
        )
