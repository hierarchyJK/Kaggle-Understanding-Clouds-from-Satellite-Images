# -*- coding:utf-8 -*-
"""
@project: JJKK
@author: KunJ
@file: Exploring_Predictions.py
@ide: Pycharm
@time: 2019-11-07 10:49:36
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

from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl.runner import SupervisedRunner
from catalyst.contrib.models.segmentation import Unet
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
        albu.HorizontalFlip(),
        albu.OneOf([
            albu.RandomBrightnessContrast(),
            albu.RandomGamma(),
        ], p=0.3),
        albu.OneOf([
            albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albu.GridDistortion(),
            albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
        albu.ShiftScaleRotate(),
        albu.Resize(320, 640, always_apply=True),
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

def dice(img1, img2): # 用于寻找最优阈值
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
valid_ids_df1 = pd.read_csv('/home/jiangkun/ShallowCloud/CV/val_1.csv')
valid_ids_df2 = pd.read_csv('/home/jiangkun/ShallowCloud/CV/val_2.csv')
valid_ids_df3 = pd.read_csv('/home/jiangkun/ShallowCloud/CV/val_3.csv')
valid_ids_df4 = pd.read_csv('/home/jiangkun/ShallowCloud/CV/val_4.csv')
valid_ids_df5 = pd.read_csv('/home/jiangkun/ShallowCloud/CV/val_5.csv')
print(valid_ids_df5.shape,valid_ids_df1.shape, valid_ids_df2.shape, valid_ids_df3.shape, valid_ids_df4.shape)
# valid_ids_df = pd.concat([valid_ids_df1, valid_ids_df2, valid_ids_df3, valid_ids_df4, valid_ids_df5])
valid_ids_df = pd.read_csv('/home/jiangkun/ShallowCloud/CV/val_1.csv')
print(valid_ids_df.shape)
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
# DEVICE = 'cuda:1'
# print(ENCODER)
# ACTIVATION = None
# model = smp.Unet(
#     encoder_name=ENCODER,
#     encoder_weights=ENCODER_WEIGHTS,
#     classes=4,
#     activation=ACTIVATION,
# )
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('可用GPU数量:',torch.cuda.device_count())
# if torch.cuda.device_count() > 1:
#     print('Let us use', torch.cuda.device_count(), 'GPUs!')
#     model = nn.DataParallel(model)

# model.to(device)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


num_workers = 4
bs = 1
valid_dataset_lst = []

valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms = get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))


valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

device = torch.device('cuda')
model_pth = [
            # '/home/jiangkun/ShallowCloud/smp/best_model_se_resnext50_32x4d.pth', #0.6445
             # '/home/jiangkun/ShallowCloud/smp/best_model_resnet50.pth',#0.6406
             # '/home/jiangkun/ShallowCloud/smp/best_model_se_resnet50.pth',
             # '/home/jiangkun/ShallowCloud/smp/best_model_efficientnet-b2.pth',

             '/home/jiangkun/ShallowCloud/smp/best_model_resnet34.pth', # 0.6522
             '/home/jiangkun/ShallowCloud/smp/best_model_FPNresnet34.pth', #0.6531
             # '/home/jiangkun/ShallowCloud/smp/best_model_FPNresnet34_CV2.pth',
             # '/home/jiangkun/ShallowCloud/smp/best_model_FPNresnet34_CV3.pth',
             # '/home/jiangkun/ShallowCloud/smp/best_model_FPNresnet34_CV4.pth',
             # '/home/jiangkun/ShallowCloud/smp/best_model_FPNresnet34_CV5.pth',
             ]
models = []
for pth in model_pth:
    model = torch.load(pth, map_location=lambda storage, loc: storage)
    model.to(device)
    model.eval()
    models.append(model)
print(len(models))
# print(model_pth[-1])
# model = torch.load(model_pth[-1], map_location=lambda storage, loc: storage)
# model.to(device)
# model.eval()

probabilities = np.zeros(shape=(len(valid_ids) * 4, 350, 525), dtype=np.float32)
encoded_pixels = []

valid_mask = []
print(len(valid_dataset))

batch_preds = 0
for i, batch in enumerate(tqdm.tqdm(valid_loader)):
    image_batch, mask_batch = batch
    # batch_preds = torch.sigmoid(model(image_batch.to(device))).detach().cpu().numpy()
    for model in models:
        batch_preds += torch.sigmoid(model(image_batch.to(device))).detach().cpu().numpy() # 这里容易bug，显示输入和网络不是同一个类型

    batch_preds /= len(models)

    for mask in mask_batch:
        for m in mask:
            m = m.numpy() # 一定要加不然报错！
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_mask.append(m)
    for idx, batch in enumerate(batch_preds):
        for j, probability in enumerate(batch):
            probability = probability
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            probabilities[i * bs * 4 + idx * 4 + j, :, :] = probability



class_params = {}
for class_id in range(4):
    print(class_id, '-'*30)
    attempts = []
    for t in range(0, 100, 5):
        t /= 100
        for ms in [10000, 15000, 20000, 25000, 30000]:
            masks = []
            for i in range(class_id, len(probabilities), 4):
                probability = probabilities[i]
                predict, num_predict = post_process(sigmoid(probability), threshold=t, min_size=ms)
                masks.append(predict)
            d = []
            for i, j in zip(masks, valid_mask[class_id::4]):
                if (i.sum() == 0) & (j.sum() == 0):
                    d.append(1)
                else:
                    d.append(dice(i, j))
            attempts.append((t, ms, np.mean(d)))
    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'minsize', 'dice'])

    attempts_df = attempts_df.sort_values('dice', ascending=False)
    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['minsize'].values[0]
    class_params[class_id] = (best_threshold, best_size)

print(class_params)
