import datetime
import logging
import time

import numpy as np
import torch
from tqdm import tqdm

from lib.data.metrics import evaluation
from lib.utils.comm import all_gather, is_main_process, synchronize


def compute_on_dataset(model, data_loader, device):
    model.eval()
    query_feats, query_ids, gallery_feats = [], [], []
    comp_mode = "comp" in data_loader.dataset.name

    for batch_data in tqdm(data_loader):
        imgs = batch_data["source_images"].to(device)
        texts = batch_data["text"].to(device)
        text_lengths = batch_data["text_lengths"].to(device)
        query_ids.extend(batch_data["meta_info"]["target_image_ids"])
        with torch.no_grad():
            query_feat = model.compose_img_text(imgs, texts, text_lengths, comp_mode)
            query_feats.append(query_feat)

    for imgs in tqdm(
        data_loader.dataset.get_all_imgs(data_loader.batch_sampler.batch_size)
    ):
        imgs = imgs.to(device)
        with torch.no_grad():
            gallery_feat = model.extract_img_feature(
                imgs, norm=True, comp_mode=comp_mode
            )
            gallery_feats.append(gallery_feat)

    query_feats = torch.cat(query_feats, dim=0)
    gallery_feats = torch.cat(gallery_feats, dim=0)
    if comp_mode:
        np.save("comp_embeddings.npy", gallery_feats.cpu().numpy())
    else:
        np.save("outfit_embeddings.npy", gallery_feats.cpu().numpy())

    results_dict = dict(
        query_ids=torch.tensor(query_ids),
        gallery_ids=torch.tensor(list(data_loader.dataset.all_img_ids.values())),
        similarity=query_feats @ gallery_feats.t(),
    )

    return results_dict


def compute_on_dataset_multiturn(model, data_loader, device):
    model.eval()
    query_feats, query_ids, task_weights = [], [], []
    gallery_comp_feats, gallery_outfit_feats = [], []

    for imgs in tqdm(
        data_loader.dataset.get_all_imgs(data_loader.batch_sampler.batch_size)
    ):
        imgs = imgs.to(device)
        with torch.no_grad():
            gallery_comp_feat, gallery_outfit_feat = model.extract_img_feature(
                imgs, norm=False
            )
            gallery_comp_feats.append(gallery_comp_feat)
            gallery_outfit_feats.append(gallery_outfit_feat)
    gallery_comp_feats = torch.cat(gallery_comp_feats, dim=0)
    gallery_outfit_feats = torch.cat(gallery_outfit_feats, dim=0)
    gallery_comp_feats_norm = model.norm_layer(gallery_comp_feats)
    gallery_outfit_feats_norm = model.norm_layer(gallery_outfit_feats)

    for batch_data in tqdm(data_loader):
        imgs = batch_data["source_images"].to(device)
        query_ids.extend(batch_data["meta_info"]["target_image_ids"])

        # For each turn
        for text, text_length in zip(batch_data["text"], batch_data["text_lengths"]):
            text = text.to(device)
            text_length = text_length.to(device)
            with torch.no_grad():
                query_feat, weights = model.compose_img_text(
                    imgs, text, text_length, return_weights=True
                )
                # Greedy search
                max_comp_idx = torch.argmax(
                    query_feat @ gallery_comp_feats_norm.t(), dim=1
                )
                max_outfit_idx = torch.argmax(
                    query_feat @ gallery_outfit_feats_norm.t(), dim=1
                )
                max_idx = weights[:, 0] * max_comp_idx + weights[:, 1] * max_outfit_idx
                max_idx = max_idx.long()
                imgs = (gallery_comp_feats[max_idx], gallery_outfit_feats[max_idx])

        query_feats.append(query_feat)
        task_weights.append(weights)

    query_feats = torch.cat(query_feats, dim=0)
    task_weights = torch.cat(task_weights, dim=0)
    comp_idx = task_weights[:, 0] == 1
    outfit_idx = task_weights[:, 1] == 1
    comp_similarity = query_feats[comp_idx] @ gallery_comp_feats_norm.t()
    outfit_similarity = query_feats[outfit_idx] @ gallery_outfit_feats_norm.t()
    similarity = torch.cat((comp_similarity, outfit_similarity), dim=0)
    query_ids = torch.tensor(query_ids)
    query_ids = torch.cat((query_ids[comp_idx], query_ids[outfit_idx]), dim=0)

    results_dict = dict(
        query_ids=query_ids,
        gallery_ids=torch.tensor(list(data_loader.dataset.all_img_ids.values())),
        similarity=similarity,
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
):
    logger = logging.getLogger("CompFashion.inference")
    dataset = data_loader.dataset
    logger.info(
        "Start evaluation on {} dataset({} images).".format(dataset.name, len(dataset))
    )

    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    start_time = time.time()

    if "turn" in dataset.name:
        predictions = compute_on_dataset_multiturn(model, data_loader, device)
    else:
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
        topk=torch.tensor([1, 5, 10, 50]),
    )
