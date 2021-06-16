import json
import os
import pickle
from collections import Counter
from glob import glob

import nltk
import numpy as np
import spacy

glove = spacy.load("en_vectors_web_lg").vocab

# Build vocab
cap_train_files = glob("../dataset/fashioniq/captions/cap*train.json")
counter = Counter()
for cap_file in cap_train_files:
    with open(cap_file, "r") as f:
        cap_data = json.load(f)
    for data in cap_data:
        captions = data["captions"]
        for caption in captions:
            # ----- Split hyphen ------------
            caption = caption.replace("-", " ").replace(".", "")
            toks = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(toks)
print(len(counter))
vocab_words, no_word = {}, []
for word, cnt in counter.items():
    if glove.has_vector(word):
        vocab_words[word] = glove.get_vector(word)
    elif cnt > 2:
        no_word.append(word)
        vocab_words[word] = np.random.normal(0, 0.3, (300,))
print(len(vocab_words), len(no_word))

# Save vocab
with open("../dataset/fashioniq/captions/glove_vecs.pkl", "wb") as f:
    pickle.dump(vocab_words, f)

# Build glove file
cap_files = glob("../dataset/fashioniq/captions/cap*.json")
for cap_file in cap_files:
    # Load data
    with open(cap_file, "r") as f:
        cap_data = json.load(f)
    # Save name
    sn = os.path.basename(cap_file).split(".")
    sn = ".".join(sn[:2] + ["glove"] + sn[2:]).replace(".json", ".pkl")
    save_file = os.path.join(os.path.dirname(cap_file), sn)
    # Process
    w2v_data = []
    for data in cap_data:
        captions = data["captions"]
        w2v = []
        for caption in captions:
            caption = caption.replace("-", " ").replace(".", "")  # Split hyphen
            toks = nltk.tokenize.word_tokenize(caption.lower())
            for word in toks:
                # ------ Drop UNK words ------------------
                if word in vocab_words:
                    w2v.append(vocab_words[word])
        w2v = np.stack(w2v)
        data["wv"] = w2v
    print(save_file)
    with open(save_file, "wb") as f:
        pickle.dump(cap_data, f)
