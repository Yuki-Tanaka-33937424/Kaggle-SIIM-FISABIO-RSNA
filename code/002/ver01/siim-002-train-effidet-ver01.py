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
from IPython.display import display
from IPython import get_ipython

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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import albumentations as A
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

# numba
import numba
from numba import jit

if 'kaggle_web_client' in sys.modules:
    ROOT_DIR = '/kaggle/'
else:
    ROOT_DIR = '/home/yuki/Kaggle-SIIM-FISABIO-RSNA'

if 'kaggle_web_client' in sys.modules:
    get_ipython().system(" pip install --no-deps '../input/timm-package/timm-0.1.26-py3-none-any.whl' > /dev/null")
    get_ipython().system(" pip install --no-deps '../input/pycocotools/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl' > /dev/null")

sys.path.append(os.path.join(ROOT_DIR, 'input/omegaconf'))

sys.path.append(os.path.join(ROOT_DIR, 'input/timm-pytorch-image-models/pytorch-image-models-master'))
import timm

sys.path.append(os.path.join(ROOT_DIR, 'input/pytorch-sam'))
from sam import SAM

sys.path.append(os.path.join(ROOT_DIR, 'input/timm-efficientdet-pytorch'))
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

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
    print(OUTPUT_DIR)

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
    print_freq = 50
    size = 256
    epochs = 10
    gradient_accumulation_steps = 1
    max_grad_norm = 10000
    seed = 42
    target_col = 'integer_label'
    n_fold = 5
    trn_fold = [0]
    train = True

    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {
            "batch_size": 64,
            "num_workers": 4,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True
        },
        "valid": {
            "batch_size": 128,
            "num_workers": 4,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False
        },
        "test": {
            "batch_size": 128,
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
    base_optimizer = "Adam"

    ######################
    # Scheduler #
    ######################
    scheduler_name = 'CosineAnnealingLR'
    scheduler_params = {
        "T_max": 10,
        "eta_min": 1e-6,
        "last_epoch": -1
    }

    ######################
    # Model #
    ######################
    model_name = "tf_efficientdet_d0"
    model_path_name = os.path.join(ROOT_DIR, 'input/efficientdet/efficientdet_d0-d92fd44f.pth')
    pretrained = True
    target_size = 2


# ====================================================
# Data Loading #
# ====================================================
def get_train_file_path(image_id):
    return f"../input/siim-covid19-resized-to-256px-jpg/train/{image_id}.jpg"


def get_test_file_path(image_id):
    return f"../input/siim-covid19-resized-to-256px-jpg/test/{image_id}.jpg"


updated_train_labels = pd.read_csv('../input/siim-covid19-updated-train-labels/updated_train_labels.csv')

updated_train_labels['jpg_path'] = updated_train_labels['id'].apply(get_train_file_path)
train = updated_train_labels.copy()

if CFG.debug:
    CFG.epochs = 3
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)


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


@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

    if dx < 0:
        return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = ((gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) + (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) - overlap_area)

    return overlap_area / union_area


@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold=0.5, form='pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):

        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue

        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)

            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx


@jit(nopython=True)
def calculate_precision(gts, preds, threshold=0.5, form='coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds=(0.5, ), form='coco') -> float:
    """Calculates image precision.
       The mean average precision at different intersection over union (IoU) thresholds.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0

    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision


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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# 今回は各bboxにラベルが割り振られているのではなく、bboxのラベルはopecity一択で、画像自体にラベルが4択で付与されているから、
# そのままbboxのラベルとして当てに行くとおかしくなる気がするけど、どうなんだろう。
class SiimDataset(Dataset):

    def __init__(self, df, transform=None):

        self.df = df
        self.image_ids = df['id'].values
        self.file_names = df['jpg_path'].values
        self.transform = transform

    def __getitem__(self, index: int):

        image, boxes, labels = self.load_image_and_boxes(index)

        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([index])

        if self.transform:
            for i in range(10):
                sample = self.transform(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels  # ここは多分opacityで良さそう。(negativeとかはbboxのラベルでは無いから)
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    break
        return image, target

    def __len__(self):
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(self.file_names[index]).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.df[self.df['id'] == image_id]
        boxes = []
        # bboxが複数ある場合はレコードで分かれているため、idが同じものは全てまとめている
        for bbox in records[['frac_xmin', 'frac_ymin', 'frac_xmax', 'frac_ymax']].values:
            bbox = np.clip(bbox, 0, 1.0)
            # fracは正規化した座標なので、今回はpascal_vocのformatに合わせる。(0~256で表す)
            temp = A.convert_bbox_from_albumentations(bbox, 'pascal_voc', image.shape[0], image.shape[1])
            boxes.append(temp)
        """
        [0: 'atypical', 1: 'indeterminate', 2: 'negative', 3: 'typical']
        """
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        return image, boxes, labels


# ====================================================
# Data Loader #
# ====================================================
def collate_fn(batch):
    return tuple(zip(*batch))


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
class CustomEfficientDet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = get_efficientdet_config(CFG.model_name)
        self.model = EfficientDet(self.config, pretrained_backbone=False)

        if CFG.pretrained:
            checkpoint = torch.load(CFG.model_path_name)
            self.model.load_state_dict(checkpoint)

        self.config.num_classes = 1
        self.config.image_size = 256

        self.model.class_net = HeadNet(self.config, num_outputs=self.config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
        self.model = DetBenchTrain(self.model, self.config)

    def forward(self, images, boxes, labels):
        return self.model(images, boxes, labels)


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
    for step, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = torch.stack(images)
        images = images.to(device).float()
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        batch_size = len(labels)
        with autocast(enabled=CFG.use_amp):
            loss, _, _ = model(images, boxes, labels)
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
    start = end = time.time()
    for step, (images, targets) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = torch.stack(images)
        images = images.to(device).float()
        boxes = [target['boxes'].to(device).float() for target in targets]
        labels = [target['labels'].to(device).float() for target in targets]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        batch_size = len(images)
        # compute loss
        with torch.no_grad():
            loss, _, _ = model(images, boxes, labels)

        losses.update(loss, batch_size)
        # record accuracy
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
    return losses.avg


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
                              **CFG.loader_params['train'],
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              **CFG.loader_params['valid'],
                              collate_fn=collate_fn)

    # ====================================================
    # model #
    # ====================================================
    model = CustomEfficientDet()
    model.to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop #
    # ====================================================
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss = valid_fn(valid_loader, model, criterion, device)

        scheduler_step(scheduler)

        # scoring
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict()},
                       os.path.join(OUTPUT_DIR, f'{CFG.model_name}_fold{fold}_best.pth')
                       )

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return best_loss.cpu().numpy()


# ====================================================
# main #
# ====================================================
def main():

    """
    Prepare: 1.train 2.test 3.submission 4.folds
    """

    if CFG.train:
        # train
        losses = []
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                loss = train_loop(folds, fold)
                losses.append(loss)
                LOGGER.info(f'========== fold: {fold} result ==========')
                LOGGER.info(f'best loss: {loss:.4f}')
        # CV result
        LOGGER.info('========== CV ==========')
        LOGGER.info(f'mean of loss: {np.mean(losses):.4f}')


if __name__ == '__main__':
    main()
