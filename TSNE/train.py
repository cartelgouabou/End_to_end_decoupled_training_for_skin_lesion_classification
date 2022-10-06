#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 00:20:23 2022

@author: arthur
"""

import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from utils import AverageMeter
import torch.nn.functional as F


def train(train_loader, model, criterion, criterion_hm, optimizer,args, epoch, device):
    model.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    total_loss_train, total_acc_train = [],[]

    curr_iter = (epoch - 1) * len(train_loader)

    for i, data in enumerate(train_loader):
        images, labels = data
        N = images.size(0)
        # print('image shape:',images.shape, 'label shape',labels.shape)
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # compute output
        if 'HML' in args.loss_type:
            if args.hml==True:
                loss = criterion_hm(outputs, labels)
            elif epoch >= args.start_hm_epoch:
                loss = criterion_hm(outputs, labels)
            else:
                loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % 100 == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
    return train_loss.avg, train_acc.avg, total_loss_train, total_acc_train

def validate(val_loader, model, criterion, criterion_hm, optimizer,args, epoch, device):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            # compute output
            if 'HML' in args.loss_type:
                if args.hml==True:
                    loss = criterion_hm(outputs, labels)
                elif epoch >= args.start_hm_epoch:
                    loss = criterion_hm(outputs, labels)
                else:
                    loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            prediction = outputs.max(1, keepdim=True)[1]

            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

            val_loss.update(loss.item())

    print('------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
    print('------------------------------------------------------------')
    return val_loss.avg, val_acc.avg

def backup(val_loader, model, epoch, device):
    model.eval()
    y_true = []
    y_pred = []
    y_score=[]
    
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            images, labels = data
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            probs = torch.softmax(outputs,dim=1)
            prediction = outputs.max(1, keepdim=True)[1]
            y_true.append(labels.cpu().numpy())
            y_pred.append(prediction.detach().cpu().numpy())
            y_score.append(probs.detach().cpu().numpy())

        y_true = np.concatenate(y_true)
        y_true_2D= F.one_hot(torch.from_numpy(y_true), num_classes=7).cpu().numpy()
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)
        val_bacc = balanced_accuracy_score(y_true, y_pred)
        val_auc=roc_auc_score(y_true_2D, y_score)
    print('------------------------------------------------------------')
    print('[epoch %d],  [val bacc %.5f] ,  [val auc %.5f]' % (epoch,  val_bacc,val_auc))
    print('------------------------------------------------------------')
    
    
    return val_bacc,val_auc,y_true,y_score