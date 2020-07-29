import numpy as np
import pandas as pd
import os, cv2
from torch.utils.data import Dataset, DataLoader
import torch, timm
from torch import nn, optim
from torchvision import transforms
from torch import tensor
import skimage.io
from functools import partial
from PIL import Image
import scipy as sp
import functools, time
import pickle
import albumentations as A

duplicates = pickle.load('duplicates.pkl')

df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
df['data_provider'] = df['data_provider'].replace({'karolinska':0, 'radbound':1})
df['gleason_score'] = df['gleason_score'].replace({'negative': '0+0'})
dic = pd.DataFrame(df['gleason_score'].value_counts()).to_dict()['gleason_score']
s = sorted(dic.keys())
df['gleason_score'] = df['gleason_score'].replace({i:s.index(i) for i in s})

idxs = []
for i, j in duplicates:
    idxs.append(df[df['image_id']==i].index)

def create_folds(n):
    print('preparing folds')
    df = df.drop(idxs)
    kfs = StratifiedKFold(df, df['isup_grade'])
    dist_train, dist_valid = [], []
    for fold, (train_ids, valid_ids) in enumerate(kfs.split(df['image_id'], df['isup_grade'])):
        train, valid = df.iloc[train_ids, :], df.iloc[valid_ids, :]
        dist_train.append(train['isup_grade'].value_counts().values)
        dist_valid.append(valid['isup_grade'].value_counts().values)
        train.to_pickle(f'../data/train_{fold}.pkl')
        valid_to_pickle(f'../data/valid_{fold}.pkl')
    print(dist_train, dist_valid)



