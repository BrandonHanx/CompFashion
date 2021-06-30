import numpy as np
import spacy
from tqdm import tqdm

from lib.utils.directory import read_json

glove = spacy.load("/home/ma-user/work/hanxiao/en_vectors_web_lg-2.3.0").vocab

vocab = read_json("train_data/fashioniq/captions/vocab.json")
glove_words = np.zeros((len(vocab) + 1, 300))
no_words = []

for word, idx in tqdm(vocab.items()):
    if glove.has_vector(word):
        glove_words[idx] = glove.get_vector(word)
    else:
        no_words.append(word)
        glove_words[idx] = np.random.normal(0, 0.3, (300,))

print("Unknown Words:", len(no_words))

np.save("train_data/fashioniq/captions/glove_vocab.npy", glove_words)
