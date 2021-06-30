import datetime
import logging
import os
import time

import torch
from tqdm import tqdm

from lib.data.metrics import evaluation
from lib.utils.comm import all_gather, is_main_process, synchronize


def compute_on_dataset(model, data_loader, device):
    model.eval()
    query_feats, query_ids, gallery_feats = [], [], []

    for batch_data in tqdm(data_loader):
        imgs = batch_data["source_images"].to(device)
        texts = batch_data["text"].to(device)
        text_lengths = batch_data["text_lengths"].to(device)
        query_ids.extend(batch_data["meta_info"]["target_image_ids"])
        with torch.no_grad():
            query_feat = model.norm_layer(
                model.compose_img_text(imgs, texts, text_lengths)
            )
            query_feats.append(query_feat)

    for imgs in tqdm(
        data_loader.dataset.get_all_imgs(data_loader.batch_sampler.batch_size)
    ):
        imgs = imgs.to(device)
        with torch.no_grad():
            gallery_feat = model.norm_layer(model.extract_img_feature(imgs))
            gallery_feats.append(gallery_feat)

    results_dict = dict(
        query_feats=query_feats,
        query_ids=query_ids,
        gallery_feats=gallery_feats,
        gallery_ids=list(data_loader.dataset.all_img_ids.values()),
    )

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    return predictions


def inference(
    model,
    data_loader,
    device="cuda",
    output_folder="",
    save_data=False,
    rerank=False,
):
    logger = logging.getLogger("CompFashion.inference")
    dataset = data_loader.dataset
    logger.info(
        "Start evaluation on {} dataset({} images).".format(dataset.name, len(dataset))
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

    return evaluation(
        predictions=predictions,
        output_folder=output_folder,
        save_data=save_data,
        topk=torch.tensor([1, 5, 10, 50]),
        rerank=rerank,
    )
