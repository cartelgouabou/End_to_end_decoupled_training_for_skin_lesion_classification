#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:12:33 2022

@author: arthur
"""
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from glob import glob
import numpy as np
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df['path'][idx])
        label = torch.tensor(int(self.df['label_idx'][idx]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label




def get_data(dataset_dir='./dataset/isic/',train_ratio=0.8,test_ratio=0.1):
    all_image_path = glob(os.path.join(dataset_dir, '*', '*.jpg'))
    if    len(all_image_path)==0 :
        raise Exception("No jpg image in the specified repository, please check the path. the remaining path you write is: {}".format(dataset_dir))
    df_data=pd.DataFrame(columns=['path','label','image_id'])
    for i in range(len(all_image_path)):
        df_data=df_data.append(pd.DataFrame({'path':[all_image_path[i]],
                                 'image_id':[os.path.basename(all_image_path[i])],
                                 'label':[os.path.basename(os.path.dirname(all_image_path[i]))]
                            }),
                           ignore_index=True
                   )
    df_data['label_idx'] = pd.Categorical(df_data['label']).codes
    if    (train_ratio+test_ratio>=1.) :
        raise Exception("specified a ratio for train and test set between 0 to 1. the remaining ratio for validation is: {}".format(1.- train_ratio+test_ratio))
    allFilenames_ack=np.array(df_data[df_data.label=='ACK'].image_id)
    allFilenames_bcc=np.array(df_data[df_data.label=='BCC'].image_id)
    allFilenames_bek=np.array(df_data[df_data.label=='BEK'].image_id)
    allFilenames_def=np.array(df_data[df_data.label=='DEF'].image_id)
    allFilenames_mel=np.array(df_data[df_data.label=='MEL'].image_id)
    allFilenames_nev=np.array(df_data[df_data.label=='NEV'].image_id)
    allFilenames_val=np.array(df_data[df_data.label=='VAL'].image_id)
    np.random.shuffle(allFilenames_ack)
    train_filenames_ack, test_filenames_ack, val_filenames_ack = np.split(allFilenames_ack,
                                                          [int(len(allFilenames_ack)* train_ratio),int(len(allFilenames_ack)*train_ratio+len(allFilenames_ack)*test_ratio)])
    np.random.shuffle(allFilenames_bcc)
    train_filenames_bcc, test_filenames_bcc, val_filenames_bcc = np.split(allFilenames_bcc,
                                                          [int(len(allFilenames_bcc)* train_ratio),int(len(allFilenames_bcc)*train_ratio+len(allFilenames_bcc)*test_ratio)])
    np.random.shuffle(allFilenames_bek)
    train_filenames_bek, test_filenames_bek, val_filenames_bek = np.split(allFilenames_bek,
                                                          [int(len(allFilenames_bek)* train_ratio),int(len(allFilenames_bek)*train_ratio+len(allFilenames_bek)*test_ratio)])
    np.random.shuffle(allFilenames_def)
    train_filenames_def, test_filenames_def, val_filenames_def = np.split(allFilenames_def,
                                                          [int(len(allFilenames_def)* train_ratio),int(len(allFilenames_def)*train_ratio+len(allFilenames_def)*test_ratio)])
    np.random.shuffle(allFilenames_mel)
    train_filenames_mel, test_filenames_mel, val_filenames_mel = np.split(allFilenames_mel,
                                                          [int(len(allFilenames_mel)* train_ratio),int(len(allFilenames_mel)*train_ratio+len(allFilenames_mel)*test_ratio)]) 
    np.random.shuffle(allFilenames_nev)
    train_filenames_nev, test_filenames_nev, val_filenames_nev = np.split(allFilenames_nev,
                                                          [int(len(allFilenames_nev)* train_ratio),int(len(allFilenames_nev)*train_ratio+len(allFilenames_nev)*test_ratio)])
    np.random.shuffle(allFilenames_val)
    train_filenames_val, test_filenames_val, val_filenames_val = np.split(allFilenames_val,
                                                          [int(len(allFilenames_val)* train_ratio),int(len(allFilenames_val)*train_ratio+len(allFilenames_val)*test_ratio)])
    global train_filenames 
    global test_filenames 
    global val_filenames

    train_filenames=np.concatenate((train_filenames_ack,train_filenames_bcc,train_filenames_bek,train_filenames_def,train_filenames_mel,train_filenames_nev,train_filenames_val))
    test_filenames=np.concatenate((test_filenames_ack,test_filenames_bcc,test_filenames_bek,test_filenames_def,test_filenames_mel,test_filenames_nev,test_filenames_val))
    val_filenames=np.concatenate((val_filenames_ack,val_filenames_bcc,val_filenames_bek,val_filenames_def,val_filenames_mel,val_filenames_nev,val_filenames_val))
    train_class_num_list=[len(train_filenames_ack),len(train_filenames_bcc),len(train_filenames_bek),len(train_filenames_def),len(train_filenames_mel),len(train_filenames_nev),len(train_filenames_val)]
    # This function identifies if an image is part of the train or test or val set.
    def get_set_type(x):
            # create a list of all the lesion_id's in the val set
            test_list = list(test_filenames)
            val_list = list(val_filenames)
            if str(x) in test_list:
                return 'test'
            elif str(x) in val_list:
                return 'val'
            else:
                return 'train'
    # identify train test and val rows
     # create a new colum that is a copy of the image_id column
    df_data['set_type']=df_data['image_id']
    # apply the function to this new column
    df_data['set_type']=df_data['set_type'].apply(get_set_type)
    # filter out train, test and val rows
    df_train = df_data[df_data['set_type'] == 'train']
    df_test = df_data[df_data['set_type'] == 'test']
    df_val = df_data[df_data['set_type'] == 'val']
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    df_val = df_val.reset_index()

    return df_train, df_test, df_val, train_class_num_list
