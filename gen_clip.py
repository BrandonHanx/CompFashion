import json

import clip
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device, jit=False)
vocab = json.load(open("train_data/fashioniq/captions/vocab.json"))
texts = vocab.keys()
texts = clip.tokenize(list(texts)).to(device)

with torch.no_grad():
    embeddings = model.encode_text(texts)

clip_vocab = np.zeros((len(texts) + 1, 512))
for text_idx, embedding in zip(vocab.values(), embeddings.cpu().numpy()):
    clip_vocab[text_idx] = embedding

np.save("train_data/fashioniq/captions/clip_vocab.npy", clip_vocab)
