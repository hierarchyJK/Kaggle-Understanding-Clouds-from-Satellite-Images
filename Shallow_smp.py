# -*- coding:utf-8 -*-
"""
@project: JJKK
@author: KunJ
@file: Shallow_clouds_interface.py
@ide: Pycharm
@time: 2019-11-01 16:47:23
@month: 十一月
"""

import os
import cv2
import collections
import time
import tqdm
from PIL import Image
from functools import partial
train_on_gpu = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu


from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(2019)

def get_img(x, folder: str = 'train_images'):
    """
    Return image based on image name and folder.
    """
    data_folder = os.path.join('/home/jiangkun/ShallowCloud/', folder)
    image_path = os.path.join(data_folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (1400, 2100)):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def get_training_augmentation():

    train_transform = [
        # albu.Resize(320, 640, always_apply=True),
        # albu.VerticalFlip(),
        # albu.HorizontalFlip(),
        # albu.Rotate(limit=20),
        # albu.GridDistortion(),

        albu.Resize(320, 640, always_apply=True),
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(),
        albu.Rotate(limit=20)

        # albu.OneOf([
        #     albu.RandomBrightnessContrast(),
        #     albu.RandomGamma(),
        # ], p=0.3),
        # albu.OneOf([
        #     # albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     albu.GridDistortion(),
        #     albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        # ], p=0.3),
        # albu.ShiftScaleRotate(),

    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(320, 640, always_apply=True)
    ]
    return albu.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())

train = pd.read_csv('/home/jiangkun/ShallowCloud/train.csv')
sub = pd.read_csv('/home/jiangkun/ShallowCloud/sample_submission.csv')
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])


sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
# id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
# train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=2019, stratify=id_mask_count['count'], test_size=0.1)

train_ids_df = pd.read_csv('/home/jiangkun/ShallowCloud/CV/train_5.csv')
valid_ids_df = pd.read_csv('/home/jiangkun/ShallowCloud/CV/val_5.csv')
train_ids = np.reshape(train_ids_df['img_id'].values, (len(train_ids_df),))
valid_ids = np.reshape(valid_ids_df['img_id'].values, (len(valid_ids_df),))

test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values


class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms = albu.Compose([albu.HorizontalFlip(), ToTensorV2()]),
                preprocessing=None):
        self.df = df
        if datatype != 'test':
            self.data_folder = "/home/jiangkun/ShallowCloud/train_images"
        else:
            self.data_folder = "/home/jiangkun/ShallowCloud/test_images"
        self.img_ids = img_ids

        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
print('FNP'+ENCODER+'cv5')
ACTIVATION = None
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=4,
    activation=ACTIVATION,
)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('可用GPU数量:',torch.cuda.device_count())
# if torch.cuda.device_count() > 1:
#     print('Let us use', torch.cuda.device_count(), 'GPUs!')
#     model = nn.DataParallel(model)

# model.to(device)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

num_workers = 4
bs = 8
train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms = get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loss = smp.utils.losses.BCEDiceLoss()
metrics = [
    smp.utils.metrics.IoUMetric(eps=1.),
    smp.utils.metrics.FscoreMetric(eps=1.)
]
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 3e-2},
    {'params': model.encoder.parameters(), 'lr': 3e-4},
])

train_epoch = smp.utils.train.TrainEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)
valid_epoch = smp.utils.train.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)
num_epochs= 80
max_score = 0
for i in range(1, num_epochs+1):
    print('\nEpoch:{}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    print(train_logs.keys())
    if max_score < valid_logs['iou']:
        max_score = valid_logs['iou']
        torch.save(model, '/home/jiangkun/ShallowCloud/smp/best_model_FPN{}_CV5.pth'.format(ENCODER))
        print('Model saved!')
    if i == 15:
        optimizer.param_groups[0]['lr'] = 3e-3
        print('Decrease decoder learning rate to 3e-3')
    if i == 30:
        optimizer.param_groups[0]['lr'] = 3e-4
        print('Decrease decoder learning rate to 3e-4')
    if i == 45:
        optimizer.param_groups[0]['lr'] = 3e-5
        print('Decrease decoder learning rate to 3e-5')
    if i == 60:
        optimizer.param_groups[0]['lr'] = 3e-6
        print('Decrease decoder learning rate to 3e-6')

