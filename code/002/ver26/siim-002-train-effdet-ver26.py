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

sys.path.append(os.path.join(ROOT_DIR, 'input/yuki-omegaconf/omegaconf'))

sys.path.append(os.path.join(ROOT_DIR, 'input/pytorch-sam'))
from sam import SAM

sys.path.append(os.path.join(ROOT_DIR, 'input/yuki-efficientdet/timm-efficientdet-pytorch'))
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchEval
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
    size = 512
    epochs = 10
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
            "batch_size": 12,
            "num_workers": 4,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True
        },
        "valid": {
            "batch_size": 24,
            "num_workers": 4,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False
        },
        "test": {
            "batch_size": 24,
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
        "lr": 2e-4,
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
        "T_max": 10,
        "eta_min": 1e-6,
        "last_epoch": -1
    }

    ######################
    # Model #
    ######################
    model_name = "tf_efficientdet_d4"
    model_pretrained_weight = os.path.join(
        ROOT_DIR, 'input/efficientdet/efficientdet_d4-5b370b7a.pth')
    target_size = 1


# ====================================================
# Data Loading #
# ====================================================
def get_train_file_path(image_id):
    return os.path.join(ROOT_DIR, f"input/siim-covid19-resized-to-512px-jpg/train/{image_id}.jpg")


def get_test_file_path(image_id):
    return os.path.join(ROOT_DIR, f"/input/siim-covid19-resized-to-512px-jpg/test/{image_id}.jpg")


train = pd.read_csv(os.path.join(ROOT_DIR, 'input/siim-covid19-updated-train-labels/updated_train_labels.csv'))
train['jpg_path'] = train['id'].apply(get_train_file_path)
train['detection_label'] = train.apply(lambda row: 0 if row[[
    'xmin', 'ymin', 'xmax', 'ymax']].values.tolist() == [0, 0, 1, 1] else 1, axis=1)
cols = ['xmin', 'ymin', 'xmax', 'ymax']
for idx, (xmin, ymin, xmax, ymax, label) in enumerate(zip(train['frac_xmin'].to_numpy(),
                                                          train['frac_ymin'].to_numpy(),
                                                          train['frac_xmax'].to_numpy(),
                                                          train['frac_ymax'].to_numpy(),
                                                          train['detection_label'].to_numpy())):
    if label == 0:
        train.loc[idx, cols] = [0, 0, 1, 1]
    else:
        bbox = [xmin, ymin, xmax, ymax]
        bbox = np.clip(bbox, 0, 1.0)
        train.loc[idx, cols] = A.convert_bbox_from_albumentations(bbox, 'pascal_voc', CFG.size, CFG.size)


if CFG.debug:
    CFG.epochs = 1
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

    valid = pd.DataFrame(ann, columns=['id', 'detection_label', 'xmin', 'ymin', 'xmax', 'ymax'])
    preds = pd.DataFrame(pred, columns=['id', 'detection_label', 'conf', 'xmin', 'ymin', 'xmax', 'ymax'])
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
            print('Number of files in detection after reduction: {}'.format(len(preds_unique)))

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

            scr, fp, tp = check_if_true_or_false_positive(annotations, detections, iou_threshold)
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
            s1 = "{:30s} | {:.6f} | {:7d}".format(label, average_precision, int(num_annotations))
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


def nms(boxes, scores, overlap=0.45, top_k=200):
    scores = torch.from_numpy(scores)
    boxes = torch.from_numpy(boxes)

    count = 0
    keep = scores.new(scores.size(0)).zero_().long()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    v, idx = scores.sort(0)

    idx = idx[-top_k:]

    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break

        idx = idx[:-1]

        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        inter = tmp_w * tmp_h

        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union

        idx = idx[IoU.le(overlap)]

    return keep.numpy(), count


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
            ToTensorV2(),
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


class SiimDataset(Dataset):

    def __init__(self, df, transform=None):

        self.df = df
        self.image_ids = df['id'].values
        self.file_names = df['jpg_path'].values
        self.transform = transform

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

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
        return image_id, image, target

    def __len__(self):
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(self.file_names[index]).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.df[self.df['id'] == image_id]
        # bboxが複数ある場合はレコードで分かれているため、idが同じものは全てまとめている
        boxes = [bbox for bbox in records[['xmin', 'ymin', 'xmax', 'ymax']].values]
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
def get_model_train():
    config = get_efficientdet_config(CFG.model_name)
    model = EfficientDet(config, pretrained_backbone=False)
    state = torch.load(CFG.model_pretrained_weight)
    model.load_state_dict(state)
    config.num_classes = 1
    config.image_size = CFG.size
    model.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(model, config)


def get_model_inference(model_path):
    config = get_efficientdet_config(CFG.model_name)
    model = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size = CFG.size
    model.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    state = torch.load(model_path)
    model.load_state_dict(state['model'])

    del state
    gc.collect()

    model = DetBenchEval(model, config)
    model.eval()
    return model.cuda()


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


def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    scaler = GradScaler(enabled=CFG.use_amp)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (image_ids, images, targets) in enumerate(train_loader):
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


def valid_fn(valid_loader, model, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluation mode
    model.train()
    start = end = time.time()
    for step, (image_ids, images, targets) in enumerate(valid_loader):
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

        losses.update(loss.item(), batch_size)
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


def make_predictions(valid_loader, model, device, score_threshold):
    predictions = []
    with torch.no_grad():
        for step, (image_ids, images, targets) in enumerate(valid_loader):
            images = torch.stack(images).float().to(device)
            outputs = model(images, image_scales=torch.tensor([1] * images.shape[0]).float().cuda())
            for i in range(images.shape[0]):
                # 画像一枚ごとにboxなどを出す
                boxes = outputs[i].detach().cpu().numpy()[:, :4]
                scores = outputs[i].detach().cpu().numpy()[:, 4]
                labels = outputs[i].detach().cpu().numpy()[:, 5]
                indexes = np.where(scores > score_threshold)[0]
                boxes = boxes[indexes]
                scores = scores[indexes]
                labels = labels[indexes]

                # NMS
                new_boxes = []
                new_scores = []
                new_labels = []
                for label in np.unique(labels):
                    idx = np.where(labels == label)
                    boxes = boxes[idx]
                    scores = scores[idx]
                    labels = labels[idx]

                    keep, count = nms(boxes, scores)

                    boxes = boxes[keep[:count]]
                    scores = scores[keep[:count]]
                    labels = labels[keep[:count]]

                    preds_sorted_idx = np.argsort(scores)[::-1]
                    boxes_sorted = boxes[preds_sorted_idx]
                    scores_sorted = scores[preds_sorted_idx]

                    new_boxes.append(boxes_sorted)
                    new_scores.append(scores_sorted)
                    new_labels.append(labels)

                boxes = np.concatenate(new_boxes, axis=0)
                scores = np.concatenate(new_scores, axis=0)
                labels = np.concatenate(new_labels, axis=0)

                # 予測がない場合はnoneクラスに入れる。
                if len(boxes) == 0:
                    predictions.append([image_ids[i], 0, 1, 0, 0, 1, 1])
                else:
                    for j in range(len(boxes)):
                        # 一枚の画像に対してboxesなどは複数あるから、それらを一つずつ入れていく
                        predictions.append([image_ids[i], 1, scores[j], boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]])
    return np.stack(predictions, axis=0)


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
    annotations = valid_folds[['id', 'detection_label', 'xmin', 'ymin', 'xmax', 'ymax']]

    train_folds_ = train_folds[train_folds[CFG.target_col] == 1].reset_index(drop=True)
    valid_folds_ = valid_folds[valid_folds[CFG.target_col] == 1].reset_index(drop=True)

    train_dataset = SiimDataset(train_folds_,
                                transform=get_transforms(data='train'))
    valid_dataset = SiimDataset(valid_folds_,
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
    model = get_model_train()
    model.to(device)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop #
    # ====================================================
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss = valid_fn(valid_loader, model, device)

        scheduler_step(scheduler)

        # scoring
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.model.state_dict()},
                       os.path.join(OUTPUT_DIR, f'{CFG.model_name}_fold{fold}_best.pth')
                       )

    # calculate CV
    model = get_model_inference(model_path=os.path.join(OUTPUT_DIR, f'{CFG.model_name}_fold{fold}_best.pth'))
    test_dataset = SiimDataset(valid_folds, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset,
                             **CFG.loader_params['valid'],
                             collate_fn=collate_fn)
    predictions = make_predictions(test_loader, model, device, score_threshold=0.01)

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return best_loss, annotations, predictions


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
        annotations = pd.DataFrame()
        predictions = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                loss, annotations_, predictions_ = train_loop(folds, fold)
                losses.append(loss)
                annotations_ = pd.DataFrame(annotations_, columns=['id', 'detection_label', 'xmin', 'ymin', 'xmax', 'ymax'])
                predictions_ = pd.DataFrame(predictions_, columns=['id', 'detection_label', 'conf', 'xmin', 'ymin', 'xmax', 'ymax'])
                annotations = pd.concat([annotations, annotations_], axis=0)
                predictions = pd.concat([predictions, predictions_], axis=0)
                mAP, AP = mean_average_precision_for_boxes(annotations_, predictions_, iou_threshold=0.5, exclude_not_in_annotations=False, verbose=True)
                LOGGER.info(f'========== fold: {fold} result ==========')
                LOGGER.info(f'best loss: {loss:.4f}')
                LOGGER.info(f'Class: opacity  AP: {AP["1"][0]:.4f}')
        if len(CFG.trn_fold) > 1:
            mAP, AP = mean_average_precision_for_boxes(annotations, predictions, iou_threshold=0.5, exclude_not_in_annotations=False, verbose=False)
            # CV result
            LOGGER.info('========== CV ==========')
            LOGGER.info(f'mean of loss: {np.mean(losses):.4f}')
            LOGGER.info(f'Class: opacity  AP: {AP["1"][0]:.4f}')

        annotations.to_pickle(os.path.join(OUTPUT_DIR, 'annotations.pkl'))
        predictions.to_pickle(os.path.join(OUTPUT_DIR, 'predictions.pkl'))


if __name__ == '__main__':
    main()
