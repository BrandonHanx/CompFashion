import json

import h5py
import numpy as np
from e2i import EmbeddingsProjector

B = "outfit"
useful_ids = np.load("useful_ids.npy").astype(int)


# paths
imgs_json = "train_data/fashionpedia/split_crop_test.json"
data_path = "data.hdf5"
output_path = f"{B}_tsne"

# get image embeddings and path to each image
embeddings = np.load(f"{B}_embeddings.npy")
embeddings = embeddings[useful_ids]
with open(imgs_json, "rt") as handle:
    name = json.load(handle)

name_list = []
for id in useful_ids:
    name_list.append(name[id])

# write an hdf5 file
with h5py.File(data_path, "w") as hf:
    hf.create_dataset("urls", data=np.asarray(name_list).astype("S"))
    hf.create_dataset("vectors", data=embeddings)
    hf.close()

# compute embeddings and create output plot
image = EmbeddingsProjector()
image.path2data = data_path
image.background_color = "WHITE"
image.method = "sklearn"
image.load_data()
image.each_img_size = 50
image.output_img_size = 2500
image.calculate_projection()
image.output_img_name = output_path
image.output_img_type = "scatter"
image.create_image()

print("done!")
