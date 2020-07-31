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
import Kornia as K
from torch.cuda import amp

duplicates = pickle.load('duplicates.pkl')
root = '../data/image_files'
mean = torch.tensor([0.0919, 0.183, 0.124])
std = torch.tensor([0.364, 0.499, 0.405])
bs = 6
sz = 256
n = 12

df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
df['data_provider'] = df['data_provider'].replace({'karolinska':0, 'radbound':1})
df['gleason_score'] = df['gleason_score'].replace({'negative': '0+0'})
dic = pd.DataFrame(df['gleason_score'].value_counts()).to_dict()['gleason_score']
s = sorted(dic.keys())
df['gleason_score'] = df['gleason_score'].replace({i:s.index(i) for i in s})

# removing the duplicate images
idxs = []
for i, j in duplicates:
    idxs.append(df[df['image_id']==i].index)

df = df.drop(idxs, inplace=True).reset_index(drop=True)

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

# preprocessing funcs

def np2float(x):
    return torch.from_numpy(np.array(x, dtype=np.float32)).permute(2,0,1).contiguous()/255.

def rgb2tensor(x):
    x = np.moveaxis(x, -1, 0)
    x = np.ascontiguousarray(x)
    return torch.from_numpy(x/255.)

def get_tiles(f, train=False):
    imgs = []
    for i in range(n):
        imgs.append(rgb2tensor(~cv2.imread(f'{root}/{f}_{i}.png')))
    random.shuffle(imgs)
    return torch.stack(imgs)

class Expand():
    def __call__(self, x):
        return x.view(-1, n, 3, sz, sz)

class Juiced:
    def __call__(self, x):
        return x.view(-1, 3, sz, sz)

class NoOp():
    def __call__(self, x):
        return x


# transforms
train_aug = transforms.Compose([
    Juiced(),
    transforms.RandomChoice([
        K.augmentation.RandomAffine(30, shear=(-3, 3), scale=(0.95, 1.1)),
        NoOp(),]),
    transforms.RandomChoice([
        K.augmentation.RandomHorizontalFlip(p=0.5),
        K.augmentation.RandomVerticalFlip(p=0.5),]),
    Expand()
    ])

aug2img = transforms.Compose([ K.tensor_to_image])

aug2img2 = transforms.Compose([
    K.augmentation.Denormalize(mean, std),
    K.tensor_to_image])

primary_tfms = transforms.Compose([K.augmentation.Normalize(mean, std)])

# Dataset classes

class PandaDataset(Dataset):
    def __init__(self, df):
        self.image_id = df.image_id.values
        self.targs = df['isup_grade'].values

    def __getitem__(self, idx):
        return primary_tfms(get_tiles(self.image_id[idx])), torch.tensor(self.targs[idx], dtype=torch.float)

    def __len__(self):
        return len(self.image_id)


# model

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x*(torch.tanh(F.softplus(x)))

def to_blur(m):
    for child_name, child in m.named_children():
        if isinstance(child, nn.MaxPool2d):
            setattr(m, child_name, K.contrib.MaxBlurPool2d(3))
        else:
            to_blur(child)

def get_base():
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
    to_blur(model)
    return model


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p= self.p, eps = self.eps)

    def __repr__(self):
        return self.__class__.__name__+'('+'p='+'{:.4f}'.format(self.p.data.tolist()[0])+','+'eps='+str(self.eps)+')'

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def calc_score(y1, y2):
    return cohen_kappa_score(y1, y2, weights='quadratic')

def accuracy(out, yb):
    return (torch.argmax(out, dim=1)==yb).float().mean()


def get_dl(df, bs=8, train=True):
    ds = PandaDataset(df)
    return DataLoader(ds, shuffle=True, batch_size=bs, num_workers=4)


class Model(nn.Module):
    def __init__(self, base, n=6):
        super().__init__()
        m = base
        self.enc = nn.Sequential(*list(m.children())[:-2])
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(), nn.Flatten(), nn.Linear(2*nc, 512),
                Mish(), nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, n))

    @amp.autocast()
    def forward(self, x):
        n = x.shape[1]
        y1 = self.enc(x.view(-1, 3, sz, sz))
        s = y1.shape
        y1 = y1.view(-1, n, s[1], s[2], s[3]).permute(0,2,1,3,4).contiguous().view(-1, s[1], s[2]*n, s[3])
        return self.head(y1)

def timer(f):
    @functools.wraps(f)
    def wrap_timer(*args, **kwargs):
        start_time = time.perf_counter()
        val = f(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f'Time elapsed: {int(run_time//60)}m {int(run_time%60)}s')
        return val
    return wrap_timer

def get_progress(idx, l, width=50, char='»', bgd = ' '):
    wrap = int((idx/l)*width)
    cover - width - wrap
    pct = idx*100/l
    return f'{char*wraps}{bgd*cover}| {pct:.2f} %'


@timer
def train_loop(dl, model, opt, scheduler, device, scaler):
    loss_acc, acc_acc = 0.0, 0.0
    preds_acc, valid_labels = [], []
    l = len(dl)
    for bi, (xb, yb) in enumerate(dl):
        xb, yb = xb.to(device), yb.to(device)
        with amp.autocast():
            preds = model(train_aug(xb).float())
            loss = loss_fn(preds, yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        opt.zero_grad()
        loss_acc += loss.item()
        acc_acc += accuracy(preds.detach(), yb.detach())
        preds_acc.append(torch.argmax(preds, 1).detach().cpu().numpy())
        valid_labels.append(yb.detach().cpu().numpy())
        print(f'\rtrain metric: {get_progress(bi, l)}, accuracy: {(100* acc_acc/(bi+1)):.3f} ↑, loss: {(loss_acc/(bi+1)):.4f} ↓', end='', file=sys.stdout, flush=True)

    score_acc = calc_score(np.concatenate(preds_acc), np.concatenate(valid_labels))
    print(f'\rtrain_metrics loss: {loss_acc/(bi+1):.5f}, score: {score_acc:.5f}, accuracy: {acc_acc/(bi+1):.5f}', end = ' ')

@timer
def valid_loop(dl, model, device):
    acc_acc, loss_acc = 0.0, 0.0
    preds_acc, valid_labels = [], []
    l = len(dl)
    with torch.no_grad():
        for bi, (xb, yb) in enumerate(dl):
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb.float())
            loss = loss_fn(preds, yb)
            loss_acc += loss.item()
            acc_acc += accuracy(preds.detach(), yb.detach())
            preds_acc.append(torch.argmax(preds, 1).detach().cpu().numpy())
            valid_labels.append(yb.detach().cpu().numpy())
            print(f'\rvalid metric: {get_progress(bi, l)}, accuracy: {(100*acc_acc/(bi+1)):.3f}, loss: {(loss_acc/(bi+1)):.4f}', end='', file=sys.stdout, flush=True)

    score_acc = calc_score(np.concatenate(preds_acc), np.concatenate(valid_labels))
    print(f'\rvalid_metrics loss: {loss_acc/(bi+1):.5f}, score: {score_acc:.5f}, acc: {acc_acc/(bi+1):.5f}', end = ' ')
    return score_acc

def run(fold):
    train, vals = pd.read_pickle(f'../data/train_{fold}.pkl'), pd.read_pickle(f'../working/valid_{fold}.pkl')
    model = get_base()
    device = 'cuda'
    model.to(device)
    opt = Over9000(model.parameters(), lr = 0.001)
    num_steps = int(trains.shape[0]/16)+1
    acc_score = -100

    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, steps_per_epoch=num_steps,epoch=16,
            cycle_momentum=False, pct_start=0.0, final_div_factor=100)

    scaler = amp.GradScaler()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    for epoch in range(16):
        print(f'epoch {epoch+1}')
        train_loop(get_dl(trains, 16), model, opt, scheduler, device, scaler)
        val_score = valid_loop(get_dl(vals, 16, False), model, device)
        if val_score>acc_score:
            torch.save(model.state_dict(), f'../model/model_{fold}.bin')
            print(f'the val_metrics increased to {acc_score}-->{val_score}\n')
            acc_score = val_score


run(0)
run(1)
run(2)
run(3)
run(4)
