import os
from collections import Counter
from glob import glob

import nltk

from lib.utils.directory import read_json, write_json


def get_tokens(caption):
    caption = (
        caption.replace("-", " ")
        .replace(".", "")
        .replace("(", " ")
        .replace(")", " ")
        .replace(",", " ")
    )
    toks = nltk.tokenize.word_tokenize(caption.lower())
    return [vocab[word] for word in toks]


# Build vocab
cap_files = glob("train_data/fashionpedia/*_triplets_*.json")
counter = Counter()
for cap_file in cap_files:
    cap_data = read_json(cap_file)
    for data in cap_data:
        caption = (
            data["caption"]
            .replace("-", " ")
            .replace(".", "")
            .replace("(", " ")
            .replace(")", " ")
            .replace(",", " ")
        )
        toks = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(toks)

print("Total Words:", len(counter))
vocab = dict(zip(counter.keys(), range(1, len(counter) + 1)))  # remain for pad

# Save vocab
write_json(vocab, "train_data/fashionpedia/vocab.json")

# Build cap file
for cap_file in cap_files:
    # Load data
    cap_data = read_json(cap_file)
    # Save name
    sn = os.path.basename(cap_file).split("_")
    sn = "_".join(sn[:2] + ["dict"] + sn[2:])
    save_file = os.path.join(os.path.dirname(cap_file), sn)
    # Process
    for data in cap_data:
        caption = data["caption"]
        data["wv"] = get_tokens(caption)
    write_json(cap_data, save_file)