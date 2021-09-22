import logging

import numpy as np
import torch


def rank(similarity, q_ids, g_ids, topk=[1, 5, 10, 50], get_mAP=True):
    max_rank = max(topk)
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        max_sim, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_ids[indices]  # q * k
    matches = pred_labels.eq(q_ids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    all_cmc = all_cmc[topk - 1]

    #     np.save("cmc.npy", matches.cpu().numpy())
    #     np.save("similarity.npy", max_sim.cpu().numpy())
    #     np.save("q_ids.npy", q_ids.cpu().numpy())
    #     np.save("g_ids.npy", g_ids.cpu().numpy())
    #     np.save("indices.npy", indices.cpu().numpy())

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100
    return all_cmc, mAP, indices


def jaccard(a_list, b_list):
    return float(len(set(a_list) & set(b_list))) / float(len(set(a_list) | set(b_list)))


def jaccard_mat(row_nn, col_nn):
    jaccard_sim = np.zeros(row_nn.shape[0], col_nn.shape[0])
    # FIXME: need optimization
    for i in range(row_nn.shape[0]):
        for j in range(col_nn.shape[0]):
            jaccard_sim[i, j] = jaccard(row_nn[i], col_nn[j])
    return torch.from_numpy(jaccard_sim)


def k_reciprocal(q_feats, g_feats, neighbor_num=10, alpha=0.3):
    qg_sim = torch.matmul(q_feats, g_feats.t())  # q * g
    gg_sim = torch.matmul(g_feats, g_feats.t())  # g * g

    qg_indices = torch.argsort(qg_sim, dim=1, descending=True)
    gg_indices = torch.argsort(gg_sim, dim=1, descending=True)

    qg_nn = qg_indices[:, :neighbor_num]  # q * n
    gg_nn = gg_indices[:, :neighbor_num]  # g * n

    jaccard_sim = jaccard_mat(qg_nn.cpu().numpy(), gg_nn.cpu().numpy())  # q * g
    jaccard_sim = jaccard_sim.to(qg_sim.device)
    similarity = (1.0 - alpha) * qg_sim + alpha * jaccard_sim
    return similarity  # q * g


def evaluation(
    predictions,
    topk,
):
    logger = logging.getLogger("CompFashion.inference")

    g_ids = predictions["gallery_ids"]
    q_ids = predictions["query_ids"]
    similarity = predictions["similarity"]

    cmc, _ = rank(similarity, q_ids, g_ids, topk, get_mAP=False)
    results = cmc.t().cpu().numpy()

    for k, result in zip(topk, results):
        logger.info("R@{}: {}".format(k, result))

    return cmc[2:]
