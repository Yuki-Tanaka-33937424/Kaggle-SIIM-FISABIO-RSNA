# ====================================================
# Library #
# ====================================================
import os
import gc
import sys
import math
import time
import random
import shutil
from requests import get
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn import model_selection

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

if 'kaggle_web_client' in sys.modules:
    ROOT_DIR = '/kaggle/'
else:
    ROOT_DIR = '/home/yuki/Kaggle-SIIM-FISABIO-RSNA'

sys.path.append(os.path.join(ROOT_DIR, 'input/timm-pytorch-image-models/pytorch-image-models-master'))
import timm

sys.path.append(os.path.join(ROOT_DIR, 'input/pytorch-sam'))
from sam import SAM

from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings('ignore')


# ====================================================
# Directory settings #
# ====================================================
if 'kaggle_web_client' in sys.modules:
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'working/')
else:
    name_code = os.path.splitext(os.path.basename(__file__))[0].split('-')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output/', name_code[1], name_code[-1])

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ====================================================
# Config #
# ====================================================
class CFG:
    ######################
    # Globals #
    ######################
    debug = False
    use_amp = False
    print_freq = 100
    size = 384
    epochs = 6
    gradient_accumulation_steps = 1
    max_grad_norm = 10000
    seed = 42
    target_col = 'detection_label'
    n_fold = 5
    trn_fold = [0]
    train = True

    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {
            "batch_size": 32,
            "num_workers": 4,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True
        },
        "valid": {
            "batch_size": 64,
            "num_workers": 4,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False
        },
        "test": {
            "batch_size": 64,
            "num_workers": 4,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False
        }
    }

    ######################
    # Split #
    ######################
    split_name = "StratifiedKFold"
    split_params = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42
    }

    ######################
    # Criterion #
    ######################
    loss_name = "BCEWithLogitsLoss"
    loss_params: dict = {}

    ######################
    # Optimizer #
    ######################
    optimizer_name = "AdamW"
    optimizer_params = {
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "amsgrad": False
    }
    # For SAM optimizer
    base_optimizer = "AdamW"

    ######################
    # Scheduler #
    ######################
    scheduler_name = 'CosineAnnealingLR'
    scheduler_params = {
        "T_max": 6,
        "eta_min": 1e-6,
        "last_epoch": -1
    }

    ######################
    # Model #
    ######################
    model_name = "tf_efficientnet_b3_ns"
    pretrained = True
    target_size = 1


# ====================================================
# Data Loading #
# ====================================================
def get_train_file_path(image_id):
    return os.path.join(ROOT_DIR, f"input/siim-covid19-resized-to-256px-jpg/train/{image_id}.jpg")


def get_test_file_path(image_id):
    return os.path.join(ROOT_DIR, f"/input/siim-covid19-resized-to-256px-jpg/test/{image_id}.jpg")


train = pd.read_csv(os.path.join(ROOT_DIR, 'input/siim-covid19-updated-train-labels/updated_train_labels.csv'))
train['detection_label'] = train.apply(lambda row: 0 if row[[
    'xmin', 'ymin', 'xmax', 'ymax']].values.tolist() == [0, 0, 1, 1] else 1, axis=1)
test = pd.read_csv(os.path.join(ROOT_DIR, 'input/siim-covid19-updated-train-labels/updated_sample_submission.csv'))

train['filepath'] = train['id'].apply(get_train_file_path)
test['filepath'] = test['id'].apply(get_test_file_path)

if CFG.debug:
    CFG.epochs = 1
    # train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)


# ====================================================
# Utils #
# ====================================================
def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score


def get_result(result_df):
    preds = result_df['preds'].values
    labels = result_df[CFG.target_col].values
    score = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.5f}')


def init_logger(log_file=os.path.join(OUTPUT_DIR, 'train.log')):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_torch(seed=CFG.seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = get_device()


# =================================================
# CV Split #
# =================================================
folds = train.copy()
Fold = model_selection.__getattribute__(CFG.split_name)(**CFG.split_params)
for n, (train_index, valid_index) in enumerate(Fold.split(folds, folds[CFG.target_col])):
    folds.loc[valid_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
print(folds.groupby(['fold', CFG.target_col]).size())


# ====================================================
# Transform #
# ====================================================
def get_transforms(*, data):

    if data == 'train':
        return Compose([
            Resize(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            ToTensorV2(),
        ])


# ====================================================
# Dataset #
# ====================================================
class SiimDataset(Dataset):
    def __init__(self, df=None, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = self.df.loc[idx, 'filepath']
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.df.loc[idx, CFG.target_col])
        return image.float(), label.float()


# ====================================================
# Scheduler #
# ====================================================
def get_scheduler(optimizer=None):
    if CFG.scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(CFG.scheduler_name)(optimizer, **CFG.scheduler_params)


def scheduler_step(scheduler=None, avg_val_loss=None):
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_val_loss)
    elif isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()
    elif isinstance(scheduler, CosineAnnealingWarmRestarts):
        scheduler.step()


# ====================================================
# Criterion #
# ====================================================
def get_criterion():
    if hasattr(nn, CFG.loss_name):
        return nn.__getattribute__(CFG.loss_name)(**CFG.loss_params)
    else:
        raise NotImplementedError


# ====================================================
# Optimizer #
# ====================================================
def get_optimizer(model: nn.Module):
    if CFG.optimizer_name == 'SAM':
        base_optimizer = optim.__getattribute__(CFG.base_optimizer_name)
        return SAM(model.parameters(), base_optimizer, **CFG.optimizer_params)
    else:
        if hasattr(optim, CFG.optimizer_name):
            return optim.__getattribute__(CFG.optimizer_name)(model.parameters(),
                                                              **CFG.optimizer_params)
        else:
            raise NotImplementedError


# ====================================================
# Model #
# ====================================================
class CustomEfficientNet(nn.Module):
    def __init__(self, model_name=CFG.model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return torch.squeeze(x)


# ====================================================
# Helper functions #
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    scaler = GradScaler(enabled=CFG.use_amp)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with autocast(enabled=CFG.use_amp):
            y_preds = model(images)
            loss = criterion(y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  .format(epoch + 1, step, len(train_loader),
                          data_time=data_time, loss=losses,
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0],
                          )
                  )
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          data_time=data_time, loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))
                          )
                  )
    predictions = np.concatenate(preds)
    return losses.avg, predictions


# ====================================================
# Train loop #
# ====================================================
def train_loop(folds, fold):

    LOGGER.info(f'========== fold: {fold} training ==========')

    # ====================================================
    # loader
    # ====================================================
    train_index = folds[folds['fold'] != fold].index
    valid_index = folds[folds['fold'] == fold].index

    train_folds = folds.loc[train_index].reset_index(drop=True)
    valid_folds = folds.loc[valid_index].reset_index(drop=True)

    train_dataset = SiimDataset(train_folds,
                                transform=get_transforms(data='train'))
    valid_dataset = SiimDataset(valid_folds,
                                transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              **CFG.loader_params['train'])
    valid_loader = DataLoader(valid_dataset,
                              **CFG.loader_params['valid'])

    # ====================================================
    # model #
    # ====================================================
    model = CustomEfficientNet(CFG.model_name, pretrained=CFG.pretrained)
    model.to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop #
    # ====================================================
    best_score = 0

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds[CFG.target_col].values

        scheduler_step(scheduler)

        # scoring
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - AUC: {score}')

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'preds': preds},
                       os.path.join(OUTPUT_DIR, f'{CFG.model_name}_fold{fold}_best.pth')
                       )

    check_point = torch.load(os.path.join(OUTPUT_DIR, f'{CFG.model_name}_fold{fold}_best.pth'))
    valid_folds['preds'] = check_point['preds']

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return valid_folds


# ====================================================
# main #
# ====================================================
def main():

    """
    Prepare: 1.train 2.test 3.submission 4.folds
    """

    if CFG.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(folds, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f'========== fold: {fold} result ==========')
                get_result(_oof_df)
        # CV result
        LOGGER.info('========== CV ==========')
        get_result(oof_df)
        # save result
        oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_df.csv'), index=False)


if __name__ == '__main__':
    main()
