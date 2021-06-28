import numpy as np
import torch

MAX_TEXT_WORDS = 30


def collate_fn(batch):
    text_feature, source, target, meta_info = zip(*batch)
    batch_size = len(text_feature)
    text_dim = len(text_feature[0])
    text_lengths = [len(x) for x in text_feature]
    max_text_words = min(max(text_lengths), MAX_TEXT_WORDS)

    text = np.zeros((batch_size, max_text_words, text_dim))
    for i, feature in enumerate(text_feature):
        text[i, : text_lengths[i], :] = feature[:max_text_words]

    source_img_ids, target_image_ids, original_captions = [], [], []
    for info in meta_info:
        source_img_ids.append(info["source_img_id"])
        target_image_ids.append(info["target_img_id"])
        original_captions.append(info["original_caption"])
    meta_info = dict(
        source_img_ids=source_img_ids,
        target_image_ids=target_image_ids,
        original_captions=original_captions,
    )

    return {
        "text": torch.FloatTensor(text),
        "text_lengths": torch.LongTensor(text_lengths),
        "source_images": torch.stack(source),
        "target_images": torch.stack(target),
        "meta_info": meta_info,
    }
