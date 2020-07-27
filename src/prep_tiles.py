import os, math
import cv2
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

df_train = pd.read_csv('../data/train.csv')
images = df['image_id'].values
train_path = '../data/train_images'
export_dir = '../data/export_images'

sz = 256
n = 64

def open_img(f):
    return skimage.io.MultiImage(os.path.join(train_path, f'.tiff'))[1]

def pad(img, h, w):
    # incase of odd number
    top_pad = np.floor((h - img.shape[0])/2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0])/2).astype(np.uint16)
    right_pad = np.floor((w - img.shape[1])/2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1])/2).astype(np.uint16)
    return np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0,0)), mode='reflect')

def get_tiles(x):
    h, w, _ = x.shape
    x = pad(x, (h//sz+1)*sz, (w//sz+1)*sz)
    x = x.reshape(x.shape[0]//sz, sz, x.shape[1]//sz, sz, 3)
    x = x.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    if len(x)<N:
        x = np.pad(x, ((0, N-len(x)), (0,0), (0,0), (0,0)), constant_values=0)
    idxs = np.argsort(x.reshape(x.shape[0], -1))[:N]
    x = x[idxs]
    return ~x

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def run():
    x_tot, x2_tot = [], []
    for name in tqdm(images):
        tiles = get_tiles(open_img(name))
        for k,t in enumerate(tiles):
            x_tot.append((t/255.0).reshape(-1, 3).mean(0))
            x2_tot.append((t/255.0)**2).reshape(-1, 3).mean(0))
            Image.fromarray(t).save(os.path.join(export_dir, f'{name}_{k}.png'))
    img_avg = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avg**2)
    print('mean:', img_avg, ', std:', np.sqrt(img_std))

run()
