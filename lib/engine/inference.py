import datetime
import logging
import os
import time

import torch
from torchvision.transforms.functional import to_pil_image
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
            query_feat = model.compose_img_text(imgs, texts, text_lengths)
            # if isinstance(query_feat, list):
            #     query_feat = torch.cat(query_feat, dim=-1)
            query_feats.append(query_feat)

    for imgs in tqdm(
        data_loader.dataset.get_all_imgs(data_loader.batch_sampler.batch_size)
    ):
        imgs = imgs.to(device)
        with torch.no_grad():
            gallery_feat = model.extract_img_feature(imgs, single=True)
            # if isinstance(gallery_feat, list):
            #     gallery_feat = torch.cat(gallery_feat, dim=-1)
            gallery_feats.append(gallery_feat)

    results_dict = dict(
        query_feats=query_feats,
        query_ids=query_ids,
        gallery_feats=gallery_feats,
        gallery_ids=list(data_loader.dataset.all_img_ids.values()),
    )

    return results_dict


def generation_on_dataset(model, data_loader, device):
    model.eval()
    tgt_imgs, pred_imgs = [], []

    for batch_data in tqdm(data_loader):
        imgs = batch_data["source_images"].to(device)
        texts = batch_data["text"].to(device)
        text_lengths = batch_data["text_lengths"].to(device)
        imgs_target = batch_data["target_images"].to(device)

        with torch.no_grad():
            pred_img, tgt_img = model.reconstruct(
                imgs, texts, text_lengths, imgs_target
            )

        tgt_imgs.append(tgt_img)
        pred_imgs.append(pred_img)
        break

    return torch.cat(pred_imgs, dim=0), torch.cat(tgt_imgs, dim=0)


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
    gen_mode=False,
):
    logger = logging.getLogger("CompFashion.inference")
    dataset = data_loader.dataset
    logger.info(
        "Start evaluation on {} dataset({} images).".format(dataset.name, len(dataset))
    )

    if gen_mode:
        pred_imgs, tgt_imgs = generation_on_dataset(model, data_loader, device)
        i = 0
        for pred_img, tgt_img in zip(pred_imgs, tgt_imgs):
            to_pil_image(pred_img).save(output_folder + "/{}_rec.png".format(i))
            to_pil_image(tgt_img).save(output_folder + "/{}_tgt.png".format(i))
            i += 1
        return

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
