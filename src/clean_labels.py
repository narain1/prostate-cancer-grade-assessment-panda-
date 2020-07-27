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
import albumentations as A
from torch.nn.parameter import Parameter
from albumentations.core.transforms_interface import DualTransform
from torch.nn.utils import spectral_norm
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
df['data_provider'] = df['data_provider'].replace({'karolinska':0, 'radbound':1})
dic = pd.DataFrame(df['gleason_s
