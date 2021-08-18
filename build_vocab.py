import os
from collections import Counter
from glob import glob

import nltk
import spacy
from spellchecker import SpellChecker

from lib.utils.directory import read_json, write_json


def get_tokens(caption):
    caption = (
        caption.replace("-", " ").replace(".", "").replace("/", " ").replace("'", " ")
    )
    toks = nltk.tokenize.word_tokenize(caption.lower())
    final_toks = []
    for tok in toks:
        if tok in correction.keys():
            final_toks.append(final_vocab[correction[tok]])
        else:
            final_toks.append(final_vocab[tok])
    return final_toks


# Build vocab
cap_files = glob("train_data/fashioniq/captions/cap*.json")
counter = Counter()
for cap_file in cap_files:
    cap_data = read_json(cap_file)
    for data in cap_data:
        captions = data["captions"]
        for caption in captions:
            # ----- Split hyphen ------------
            caption = (
                caption.replace("-", " ")
                .replace(".", "")
                .replace("/", " ")
                .replace("'", " ")
            )
            toks = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(toks)

print("Total Words:", len(counter))
vocab = dict(zip(counter.keys(), range(5, len(counter) + 6)))  # remain for other tokens
vocab["<NULL>"] = 0
vocab["<UNK>"] = 1
vocab["<START>"] = 2
vocab["<END>"] = 3
vocab["<LINK>"] = 4

# Save vocab
write_json(vocab, "train_data/fashioniq/captions/vocab_nocheck.json")

# Filter unknown words
glove = spacy.load("en_vectors_web_lg").vocab
spell = SpellChecker()
unknown_words = []
for word, idx in vocab.items():
    if idx < 5:
        continue
    if not glove.has_vector(word):
        unknown_words.append(word)

print("Unknown Words:", len(unknown_words))
misspelled = spell.unknown(unknown_words)
correction = dict()
for word in misspelled:
    correction[word] = spell.correction(word)
write_json(correction, "train_data/fashioniq/captions/correction.json")

# Merge correction
final_words = []
for word, idx in vocab.items():
    if idx < 5:
        continue
    if word in correction.keys():
        final_words.append(correction[word])
    else:
        final_words.append(word)
final_words = list(set(final_words))

print("Final Words:", len(final_words))
final_vocab = dict(
    zip(final_words, range(5, len(final_words) + 6))
)  # remain for other tokens
final_vocab["<NULL>"] = 0
final_vocab["<UNK>"] = 1
final_vocab["<START>"] = 2
final_vocab["<END>"] = 3
final_vocab["<LINK>"] = 4
write_json(final_vocab, "train_data/fashioniq/captions/vocab.json")


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
        data["wv"] = (
            get_tokens(captions[0]) + [vocab["<LINK>"]] + get_tokens(captions[1])
        )
    write_json(cap_data, save_file)
