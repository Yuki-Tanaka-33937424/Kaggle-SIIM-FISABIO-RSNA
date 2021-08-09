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

import albumentations as A
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, HueSaturationValue, CoarseDropout)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

if 'kaggle_web_client' in sys.modules:
    ROOT_DIR = '/kaggle/'
else:
    ROOT_DIR = '/home/yuki/Kaggle-SIIM-FISABIO-RSNA'

sys.path.append(os.path.join(ROOT_DIR, 'input/timm-new'))
import timm_new

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
    target_cols = ['Negative for Pneumonia', 'Typical Appearance',
                   'Indeterminate Appearance', 'Atypical Appearance']
    n_fold = 5
    trn_fold = [0]
    train = True

    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {
            "batch_size": 16,
            "num_workers": 4,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True
        },
        "valid": {
            "batch_size": 32,
            "num_workers": 4,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False
        },
        "test": {
            "batch_size": 32,
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
    split_col = 'split_label'
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
    optimizer_name = "SAM"
    optimizer_params = {
        "lr": 5e-6,
        "weight_decay": 1e-6,
        "amsgrad": False
    }
    # For SAM optimizer
    base_optimizer_name = "AdamW"

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
    model_name = "resnet200d"
    student_dir = os.path.join(ROOT_DIR, 'output/016/ver01')
    pretrained = True
    target_size = 4


# ====================================================
# Data Loading #
# ====================================================
def get_train_file_path(image_id):
    return os.path.join(ROOT_DIR, f"input/siimcovid19-512-img-png-600-study-png/study/{image_id}.png")


def get_test_file_path(image_id):
    # テストデータのパスはこれではないが、今回は読み込まないのでこのままにしておく
    return os.path.join(ROOT_DIR, f"input/siimcovid19-512-img-png-600-study-png/study/{image_id}.png")


train_study = pd.read_csv(os.path.join(ROOT_DIR, 'input/siim-covid19-detection/train_study_level.csv'))
train_study['study_id'] = train_study['id'].apply(lambda x: x.split('_')[0])
train_annotation = pd.read_csv(os.path.join(ROOT_DIR, 'input/siim-covid19-updated-train-labels/updated_train_labels.csv'))
test = pd.read_csv(os.path.join(ROOT_DIR, 'input/siim-covid19-detection/sample_submission.csv'))

train_annotation['detection_label'] = train_annotation.apply(lambda row: 0 if row[[
    'xmin', 'ymin', 'xmax', 'ymax']].values.tolist() == [0, 0, 1, 1] else 1, axis=1)
cols = ['xmin', 'ymin', 'xmax', 'ymax']
for idx, (xmin, ymin, xmax, ymax, label) in enumerate(zip(train_annotation['frac_xmin'].to_numpy(),
                                                          train_annotation['frac_ymin'].to_numpy(),
                                                          train_annotation['frac_xmax'].to_numpy(),
                                                          train_annotation['frac_ymax'].to_numpy(),
                                                          train_annotation['detection_label'].to_numpy())):
    if label == 0:
        train_annotation.loc[idx, cols] = [0, 0, 1, 1]
    else:
        bbox = [xmin, ymin, xmax, ymax]
        train_annotation.loc[idx, cols] = A.convert_bbox_from_albumentations(
            bbox, 'pascal_voc', 600, 600)

train_study['split_label'] = train_study[CFG.target_cols].apply(lambda row: row.values.argmax(), axis=1)

train_study['filepath'] = train_study['id'].apply(get_train_file_path)
test['filepath'] = test['id'].apply(get_test_file_path)

# ====================================================
# External Data #
# ====================================================
extradata = pd.read_csv(os.path.join(ROOT_DIR, 'input/ricord-covid19-xray-positive-tests/MIDRC-RICORD-meta.csv'))
extradata = extradata.dropna(subset=['labels'])


def locate_row_to_delete(a):
    if a.max() <= 0.5:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    else:
        return a


def encode_labels(df):
    df = df[['fname', 'labels']]

    #initialize label columns
    df['Negative for Pneumonia'] = 0
    df['Typical Appearance'] = 0
    df['Indeterminate Appearance'] = 0
    df['Atypical Appearance'] = 0

    #Count occurences of each category
    df['Negative for Pneumonia'] = df.labels.apply(lambda x: x.count('Negative'))
    df['Typical Appearance'] = df.labels.apply(lambda x: x.count('Typical'))
    df['Indeterminate Appearance'] = df.labels.apply(lambda x: x.count('Indeterminate'))
    df['Atypical Appearance'] = df.labels.apply(lambda x: x.count('Atypical'))

    #df to array for computations
    labels_np = df[['Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']].values
    temp = labels_np / labels_np.sum(axis=1, keepdims=True)
    temp = np.apply_along_axis(locate_row_to_delete, 1, temp)

    temp -= 0.51

    temp[temp > 0] = 1
    temp[temp < 0] = 0

    df['Negative for Pneumonia'] = temp[:, 0]
    df['Typical Appearance'] = temp[:, 1]
    df['Indeterminate Appearance'] = temp[:, 2]
    df['Atypical Appearance'] = temp[:, 3]

    df.dropna(subset=['Negative for Pneumonia'], inplace=True)
    return df


output = encode_labels(extradata)
output = output.rename(columns={'fname': 'id'})
nan_value = float("NaN")
output.replace("", nan_value, inplace=True)
output.dropna(subset=["labels"], inplace=True)
output = output.drop(['labels'], axis=1)
output['split_label'] = output[CFG.target_cols].apply(lambda row: row.values.argmax(), axis=1)
output['filepath'] = output['id'].apply(lambda x: os.path.join(ROOT_DIR, 'input/ricord-covid19-xray-positive-tests/MIDRC-RICORD/MIDRC-RICORD', x))
for col in CFG.target_cols:
    output[col] = output[col].apply(lambda x: int(x))
output = output.reset_index(drop=True)


if CFG.debug:
    CFG.epochs = 1
    train_study = train_study.sample(n=300, random_state=CFG.seed).reset_index(drop=True)


# ====================================================
# Utils #
# ====================================================
def get_annotations(df, col):
    df_ = df.copy()
    df_ = df_[['id', col]]
    df_ = df_.rename(columns={col: 'detection_label'})
    df_bbox = pd.DataFrame({
        'xmin': [0] * len(df_),
        'ymin': [0] * len(df_),
        'xmax': [1] * len(df_),
        'ymax': [1] * len(df_),
    })
    df_ = pd.concat([df_, df_bbox], axis=1)
    return df_


def get_predictions(df, col):
    df_ = df.copy()
    df_ = df_[['id', f'pred_{col}']]
    df_ = df_.rename(columns={f'pred_{col}': 'conf'})
    df_bbox = pd.DataFrame({
        'detection_label': ['1'] * len(df_),
        'xmin': [0] * len(df_),
        'ymin': [0] * len(df_),
        'xmax': [1] * len(df_),
        'ymax': [1] * len(df_),
    })
    df_ = pd.concat([df_, df_bbox], axis=1)
    return df_


def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:, i], y_pred[:, i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores


def get_result(result_df):
    preds = result_df[[f'pred_{c}' for c in CFG.target_cols]].values
    labels = result_df[CFG.target_cols].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}  Scores: {np.round(scores, decimals=4)}')


def compute_overlap(boxes, query_boxes):
    """
    Args
        boxes:       (N, 4) ndarray of float
        query_boxes: (4)    ndarray of float
    Returns
        overlaps: (N) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    overlaps = np.zeros((N), dtype=np.float64)
    box_area = (
        (query_boxes[2] - query_boxes[0]) *
        (query_boxes[3] - query_boxes[1])
    )
    for n in range(N):
        iw = (
            min(boxes[n, 2], query_boxes[2]) -
            max(boxes[n, 0], query_boxes[0])
        )
        if iw > 0:
            ih = (
                min(boxes[n, 3], query_boxes[3]) -
                max(boxes[n, 1], query_boxes[1])
            )
            if ih > 0:
                ua = np.float64(
                    (boxes[n, 2] - boxes[n, 0]) *
                    (boxes[n, 3] - boxes[n, 1]) +
                    box_area - iw * ih
                )
                overlaps[n] = iw * ih / ua
    return overlaps


def check_if_true_or_false_positive(annotations, detections, iou_threshold):
    annotations = np.array(annotations, dtype=np.float64)
    scores = []
    false_positives = []
    true_positives = []
    # a GT box should be mapped only one predicted box at most.
    detected_annotations = []
    for d in detections:
        scores.append(d[4])
        if len(annotations) == 0:
            false_positives.append(1)
            true_positives.append(0)
            continue
        overlaps = compute_overlap(annotations, d[:4])
        assigned_annotation = np.argmax(overlaps)
        max_overlap = overlaps[assigned_annotation]
        if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
            false_positives.append(0)
            true_positives.append(1)
            detected_annotations.append(assigned_annotation)
        else:
            false_positives.append(1)
            true_positives.append(0)
    return scores, false_positives, true_positives


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_real_annotations(table):
    res = dict()
    ids = table['id'].values.astype(np.str)
    labels = table['detection_label'].values.astype(np.str)
    xmin = table['xmin'].values.astype(np.float32)
    xmax = table['xmax'].values.astype(np.float32)
    ymin = table['ymin'].values.astype(np.float32)
    ymax = table['ymax'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xmin[i], ymin[i], xmax[i], ymax[i]]
        res[id][label].append(box)

    return res


def get_detections(table):
    res = dict()
    ids = table['id'].values.astype(np.str)
    labels = table['detection_label'].values.astype(np.str)
    scores = table['conf'].values.astype(np.float32)
    xmin = table['xmin'].values.astype(np.float32)
    xmax = table['xmax'].values.astype(np.float32)
    ymin = table['ymin'].values.astype(np.float32)
    ymax = table['ymax'].values.astype(np.float32)

    for i in range(len(ids)):
        id = ids[i]
        label = labels[i]
        if id not in res:
            res[id] = dict()
        if label not in res[id]:
            res[id][label] = []
        box = [xmin[i], ymin[i], xmax[i], ymax[i], scores[i]]
        res[id][label].append(box)
    return res


def mean_average_precision_for_boxes(ann, pred, iou_threshold=0.5, exclude_not_in_annotations=False, verbose=True):
    """
    :param ann: path to CSV-file with annotations or numpy array of shape (N, 6)
    :param pred: path to CSV-file with predictions (detections) or numpy array of shape (N, 7)
    :param iou_threshold: IoU between boxes which count as 'match'. Default: 0.5
    :param exclude_not_in_annotations: exclude image IDs which are not exist in annotations. Default: False
    :param verbose: print detailed run info. Default: True
    :return: tuple, where first value is mAP and second values is dict with AP for each class.
    """

    valid = pd.DataFrame(
        ann, columns=['id', 'detection_label', 'xmin', 'ymin', 'xmax', 'ymax'])
    preds = pd.DataFrame(
        pred, columns=['id', 'detection_label', 'conf', 'xmin', 'ymin', 'xmax', 'ymax'])
    ann_unique = valid['id'].unique()
    preds_unique = preds['id'].unique()

    if verbose:
        print('Number of files in annotations: {}'.format(len(ann_unique)))
        print('Number of files in predictions: {}'.format(len(preds_unique)))

    # Exclude files not in annotations!
    if exclude_not_in_annotations:
        preds = preds[preds['id'].isin(ann_unique)]
        preds_unique = preds['id'].unique()
        if verbose:
            print('Number of files in detection after reduction: {}'.format(
                len(preds_unique)))

    unique_classes = valid['detection_label'].unique().astype(np.str)
    if verbose:
        print('Unique classes: {}'.format(len(unique_classes)))

    all_detections = get_detections(preds)
    all_annotations = get_real_annotations(valid)
    if verbose:
        print('Detections length: {}'.format(len(all_detections)))
        print('Annotations length: {}'.format(len(all_annotations)))

    average_precisions = {}
    for zz, label in enumerate(sorted(unique_classes)):

        # Negative class
        if str(label) == 'nan':
            continue

        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0.0

        for i in range(len(ann_unique)):
            detections = []
            annotations = []
            id = ann_unique[i]
            if id in all_detections:
                if label in all_detections[id]:
                    detections = all_detections[id][label]
            if id in all_annotations:
                if label in all_annotations[id]:
                    annotations = all_annotations[id][label]

            if len(detections) == 0 and len(annotations) == 0:
                continue

            num_annotations += len(annotations)

            scr, fp, tp = check_if_true_or_false_positive(
                annotations, detections, iou_threshold)
            scores += scr
            false_positives += fp
            true_positives += tp

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        false_positives = np.array(false_positives)
        true_positives = np.array(true_positives)
        scores = np.array(scores)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations, precision, recall
        if verbose:
            s1 = "{:30s} | {:.6f} | {:7d}".format(
                label, average_precision, int(num_annotations))
            print(s1)

    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations, _, _) in average_precisions.items():
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision
    mean_ap = precision / present_classes
    if verbose:
        print('mAP: {:.6f}'.format(mean_ap))
    return mean_ap, average_precisions


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
folds = train_study.copy()
folds_extra = output.copy()
Fold = model_selection.__getattribute__(CFG.split_name)(**CFG.split_params)
for n, (train_index, valid_index) in enumerate(Fold.split(folds, folds[CFG.split_col])):
    folds.loc[valid_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)
for n, (train_index, valid_index) in enumerate(Fold.split(folds_extra, folds_extra[CFG.split_col])):
    folds_extra.loc[valid_index, 'fold'] = int(n)
folds_extra['fold'] = folds_extra['fold'].astype(int)
folds = pd.concat([folds, folds_extra], axis=0).reset_index(drop=True)
print(folds.groupby(['fold', CFG.split_col]).size())


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
            ShiftScaleRotate(p=0.2),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.2),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.2),
            CoarseDropout(p=0.2),
            Cutout(p=0.2),
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
        image = cv2.imread(filepath).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transform:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.df.loc[idx, CFG.target_cols])
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
class CustomResNet200D(nn.Module):
    def __init__(self, model_name=CFG.model_name, pretrained=False):
        super().__init__()
        self.model = timm_new.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return features, pooled_features, torch.squeeze(output)


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
            _, _, y_preds = model(images)
            loss = criterion(y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            if CFG.optimizer_name != 'SAM':
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                optimizer.first_step(zero_grad=True)
                _, _, y_preds_ = model(images)
                criterion(y_preds_, labels).backward()
                optimizer.second_step(zero_grad=True)
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
            _, _, y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.softmax(1).to('cpu').numpy())
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
    model = CustomResNet200D(CFG.model_name, pretrained=CFG.pretrained)
    state = torch.load(os.path.join(CFG.student_dir, f'{CFG.model_name}_fold{fold}_best.pth'))
    model.load_state_dict(state['model'])
    model.to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop #
    # ====================================================
    best_score = -np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds[CFG.target_cols].values

        scheduler_step(scheduler)

        # scoring
        for c in [f'pred_{c}' for c in CFG.target_cols]:
            valid_folds[c] = np.nan
        valid_folds[[f'pred_{c}' for c in CFG.target_cols]] = preds
        mAPs = []
        for col in CFG.target_cols:
            annotations = get_annotations(valid_folds, col)
            predictions = get_predictions(valid_folds, col)
            mAP, AP = mean_average_precision_for_boxes(annotations, predictions, iou_threshold=0.5, exclude_not_in_annotations=False, verbose=False)
            mAPs.append(AP["1"][0])
        score = np.mean(mAPs)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Negative: {mAPs[0]:.4f}  Typical: {mAPs[1]:.4f}  Indeterminate: {mAPs[2]:.4f}  Atypical: {mAPs[3]:.4f}')
        LOGGER.info(f'Epoch {epoch+1} - mAP: {score}')

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'preds': preds},
                       os.path.join(OUTPUT_DIR, f'{CFG.model_name}_fold{fold}_best.pth')
                       )

    check_point = torch.load(os.path.join(OUTPUT_DIR, f'{CFG.model_name}_fold{fold}_best.pth'))
    for c in [f'pred_{c}' for c in CFG.target_cols]:
        valid_folds[c] = np.nan
    valid_folds[[f'pred_{c}' for c in CFG.target_cols]] = check_point['preds']

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
                oof_df = pd.concat([oof_df, _oof_df]).reset_index(drop=True)
                LOGGER.info(f'========== fold: {fold} result ==========')
                mAPs = []
                for col in CFG.target_cols:
                    annotations_ = get_annotations(_oof_df, col)
                    predictions_ = get_predictions(_oof_df, col)
                    mAP, AP = mean_average_precision_for_boxes(annotations_, predictions_, iou_threshold=0.5, exclude_not_in_annotations=False, verbose=False)
                    mAPs.append(AP["1"][0])
                    LOGGER.info(f'Class: {col}  AP: {AP["1"][0]:.4f}')
                LOGGER.info(f'mAP: {np.mean(mAPs):.4f}')

        # CV result
        if len(CFG.trn_fold) != 1:
            LOGGER.info('========== CV ==========')
            mAPs = []
            for col in CFG.target_cols:
                annotations = get_annotations(oof_df, col)
                predictions = get_predictions(oof_df, col)
                mAP, AP = mean_average_precision_for_boxes(annotations, predictions, iou_threshold=0.5, exclude_not_in_annotations=False, verbose=False)
                mAPs.append(AP['1'][0])
                LOGGER.info(f'Class: {col} AP: {AP["1"][0]:.4f}')
            LOGGER.info(f'mAP: {np.mean(mAPs):.4f}')

        # save result
        oof_df.to_pickle(os.path.join(OUTPUT_DIR, 'oof_df.pkl'))


if __name__ == '__main__':
    main()
