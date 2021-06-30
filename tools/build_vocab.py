import os
from collections import Counter
from glob import glob

import nltk

from lib.utils.directory import read_json, write_json

# Build vocab
cap_files = glob("train_data/fashioniq/captions/cap*.json")
counter = Counter()
for cap_file in cap_files:
    cap_data = read_json(cap_file)
    for data in cap_data:
        captions = data["captions"]
        for caption in captions:
            # ----- Split hyphen ------------
            caption = caption.replace("-", " ").replace(".", "")
            toks = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(toks)
counter.update("and")
print("Total Words:", len(counter))

vocab = dict(zip(counter.keys(), range(1, len(counter) + 1)))  # remain for pad

# Save vocab
write_json(counter, "train_data/fashioniq/captions/vocab.json")

# Build cap file
for cap_file in cap_files:
    # Load data
    cap_data = read_json(cap_file)
    # Save name
    sn = os.path.basename(cap_file).split(".")
    sn = ".".join(sn[:2] + ["dict"] + sn[2:])
    save_file = os.path.join(os.path.dirname(cap_file), sn)
    # Process
    for data in cap_data:
        captions = data["captions"]
        w2v = []
        for caption in captions:
            caption = caption.replace("-", " ").replace(".", "")  # Split hyphen
            toks = nltk.tokenize.word_tokenize(caption.lower())
            for word in toks:
                w2v.append(vocab[word])
        data["wv"] = w2v
    write_json(cap_data, save_file)
