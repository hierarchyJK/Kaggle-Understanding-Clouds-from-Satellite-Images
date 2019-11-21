# -*- coding:utf-8 -*-
"""
@project: JJKK
@author: KunJ
@file: Shallow_clould_CV.py
@ide: Pycharm
@time: 2019-11-03 22:55:22
@month: 十一月
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('/home/jiangkun/ShallowCloud/train.csv')
sub = pd.read_csv('/home/jiangkun/ShallowCloud/sample_submission.csv')
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])


# sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
# sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})

print(id_mask_count)
X = id_mask_count[['img_id']].values
y = id_mask_count[['count']].values
print(X)
print(y)
skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
skf.get_n_splits(X, y)
i = 1
for train_index, val_index in skf.split(X, y):
    print("Train_index:", train_index)
    print("Val_index:", val_index)


    Train_df = id_mask_count.loc[train_index]
    Train_df.to_csv('/home/jiangkun/ShallowCloud/CV/train_{}.csv'.format(i), index=False)
    print(Train_df.shape)
    Val_df = id_mask_count.loc[val_index]
    Val_df.to_csv('/home/jiangkun/ShallowCloud/CV/val_{}.csv'.format(i), index=False)
    print(Val_df.shape)
    i += 1
    print('-'*30)
