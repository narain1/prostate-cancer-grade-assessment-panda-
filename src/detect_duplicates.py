import numpy as np
import pandas as pd
import os, torch
import cv2
import imagehash
from tqdm.auto import tqdm
import pickle

train_image_path = '../data/train_images'

funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash]

train_fs = []

for p,d,fs in os.walk(train_image_path):
    for f in fs:
        train_fs.append(f)

acc_hashes = []

for f in tqdm(train_fs):
    img = cv2.imread(os.path.join(train_image_path, f))
    acc_hashes.append(np.array([func(image).hash for func in funcs]).reshape(256))

hashes = torch.Tensor(np.array(hashes).astype(int)).cuda()

# calculating similarity scores
sims = np.array([hashes[i]==hashes).sum(dim=1).cpu().numpy()/256 for i in range(hashes.shape[0])])


thresh = 0.9
duplicates = np.where(sims > thresh)

pairs = []

for i,j in zip(*duplicates):
    if i==j:
        continue

    pairs.append((train_fs[i], train_fs[j]))

with open('duplicates.pkl', 'wb') as f:
    pickle.dump(pairs, f)
