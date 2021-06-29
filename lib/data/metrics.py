import logging
import os

import numpy as np
import torch


def rank(similarity, q_ids, g_ids, topk=[1, 5, 10, 50], get_mAP=True):
    max_rank = max(topk)
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_ids[indices]  # q * k
    matches = pred_labels.eq(q_ids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100
    return all_cmc, mAP, indices


def evaluation(
    predictions,
    output_folder,
    topk,
    save_data=True,
):
    logger = logging.getLogger("CompFashion.inference")
    data_dir = os.path.join(output_folder, "inference_data.npz")

    if predictions is None:
        inference_data = np.load(data_dir)
        logger.info("Load inference data from {}".format(data_dir))
        g_ids = torch.tensor(inference_data["g_ids"])
        q_ids = torch.tensor(inference_data["q_ids"])
        g_feats = torch.tensor(inference_data["g_feats"])
        q_feats = torch.tensor(inference_data["q_feats"])
        similarity = torch.tensor(inference_data["similarity"])
    else:
        g_ids = torch.tensor(predictions["gallery_ids"])
        q_ids = torch.tensor(predictions["query_ids"])
        g_feats = torch.cat(predictions["gallery_feats"], dim=0)
        q_feats = torch.cat(predictions["query_feats"], dim=0)

        similarity = torch.matmul(q_feats, g_feats.t())

        if save_data:
            np.savez(
                data_dir,
                g_ids=g_ids.cpu().numpy(),
                q_ids=q_ids.cpu().numpy(),
                similarity=similarity.cpu().numpy(),
                g_feats=g_feats.cpu().numpy(),
                q_feats=q_feats.cpu().numpy(),
            )

    topk = torch.tensor(topk)

    cmc, _ = rank(similarity, q_ids, g_ids, topk, get_mAP=False)
    results = cmc.t().cpu().numpy()
    for k, result in zip(topk, results):
        logger.info("R@{}: {}".format(k, result))

    return cmc[2]
