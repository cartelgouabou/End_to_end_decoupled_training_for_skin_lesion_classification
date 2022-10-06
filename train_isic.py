#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 00:21:59 2022

@author: arthur
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.ion()   # interactive mode
from tqdm import tqdm
import os
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, models
from torch.utils.data import DataLoader
from utils import  compute_img_mean_std,prepare_folders,eval_distribution
from data_utils import get_data,CustomImageDataset
from train import train, validate,backup
from visualize import plot_confusion_matrix,learning_curve
from torch.optim.lr_scheduler import CyclicLR
from losses import LDAMLoss, FocalLoss, IBLoss, IB_FocalLoss, SoftmaxHardMiningLoss1,SoftmaxHardMiningLoss2,DynamicalMiningLoss1,DynamicalMiningLoss2
# sklearn libraries
from sklearn.metrics import classification_report
import warnings
from tensorboardX import SummaryWriter
from opts import parser
args = parser.parse_args()
device = torch.device("cpu")

root_path=os.getcwd()



if args.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# '''


#df_train, df_test, df_val, cls_num_list = get_data(root_path+args.dataset_dir)

#df_train.to_csv(root_path+'/base/df_train.csv')
#df_test.to_csv(root_path+'/base/df_test.csv')
#df_val.to_csv(root_path+'/base/df_val.csv')
#cls_num_list_pd=pd.DataFrame(cls_num_list)
#cls_num_list_pd.to_csv(root_path+'/base/cls_num_list_pd.csv')

df_train=pd.read_csv(root_path+'/base/df_train.csv')
df_train=df_train.drop(columns='Unnamed: 0')
df_test=pd.read_csv(root_path+'/base/df_test.csv')
df_test=df_test.drop(columns='Unnamed: 0')
df_val=pd.read_csv(root_path+'/base/df_val.csv')
df_val=df_val.drop(columns='Unnamed: 0')


cls_num_list= pd.read_csv(root_path+'/base/cls_num_list_pd.csv')
cls_num_list=cls_num_list.drop(columns='Unnamed: 0')
cls_num_list=cls_num_list['0']
cls_num_list=cls_num_list.tolist()

model=torch.load(root_path+'/model/efficientnetb3_model.pth')
print('[INFO]: Set all trainable layers to TRUE...')
for params in model.parameters():
    params.requires_grad = True
    
num_trainable_layers=0
trainable_layers_names=[]
for name,param in model.named_parameters():
    if param.requires_grad :
        num_trainable_layers+=1
        trainable_layers_names.append(name)
print('Number of trainable layer is:',num_trainable_layers)
        
params=model.state_dict()
keys=list(params.keys())
num_layers=len(keys)
num_freeze_layer=((100-args.ratio_finetune)*num_trainable_layers)//100
freeze_layers_names=trainable_layers_names[:int(num_freeze_layer)]


if args.weighting_type == 'None':
            per_cls_weights = None 
elif args.weighting_type == 'CS':
    per_cls_weights = 1.0 / np.array(cls_num_list)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
elif args.weighting_type == 'CB':
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
elif args.weighting_type == 'CW':
    per_cls_weights =  np.sum(cls_num_list)/cls_num_list
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
else:
    warnings.warn('Weighting type is not listed')
    

        
input_size = 300
train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.59451115, 0.59429777, 0.59418184],
                                                            [0.14208533, 0.18548788, 0.20363748])])
# define the transformation of the val images.
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.59451115, 0.59429777, 0.59418184],
                                                            [0.14208533, 0.18548788, 0.20363748])])

# Define the training set using the table train_df and using our defined transitions (train_transform)
training_set = CustomImageDataset(df_train, transform=train_transform)
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
# Same for the validation set:
validation_set = CustomImageDataset(df_val, transform=val_transform)
val_loader = DataLoader(validation_set, batch_size=args.batch_size//2, shuffle=False, num_workers=args.workers)
# Same for the test set:
test_set = CustomImageDataset(df_test, transform=val_transform)
test_loader = DataLoader(test_set, batch_size=args.batch_size//2, shuffle=False, num_workers=args.workers)

run_list=[]
for run in range(args.num_runs):
    run_list.append('run'+str(run))
args.run_list = run_list
print('curent case:')
print(args.loss_type)
print('List of run:')
print(args.run_list)
if 'SHML' in args.loss_type:
    args.store_name = '_'.join(['isic2018', 'efficientb3', args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune),'D',str(args.delta)])
elif 'CHML' in args.loss_type:
    args.store_name = '_'.join(['isic2018', 'efficientb3', args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune),'D',str(args.delta),'start_hm_type',args.hm_delay_type])
elif 'DHML' in args.loss_type:
    args.store_name = '_'.join(['isic2018', 'efficientb3', args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune),'D',str(args.delta),'TH',str(args.max_thresh)])
else:
    args.store_name = '_'.join(['isic2018', 'efficientb3', args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)])
prepare_folders(args)
args.model_path=root_path+args.model_path
args.result_dir =root_path+args.result_dir
args.history_path=root_path+args.history_path
with open(os.path.join(args.history_path, args.store_name, 'args_history.txt'), 'w') as f:
        f.write(str(args))
tf_writer = SummaryWriter(log_dir=os.path.join(args.history_path, args.store_name))
result_save=pd.DataFrame(columns=['run','best_epoch','bacc','auc'])   
args.actual_epoch=0
for run in run_list:
    print('curent run:')
    print(run)
    args.hml=False
    # model = models.efficientnet_b3(pretrained=True)
    # model.classifier[1]=nn.Linear(1536, 7)
    model=torch.load(root_path+'/model/efficientnetb3_model.pth')
    print('[INFO]: Set all trainable layers to TRUE...')
    for params in model.parameters():
        params.requires_grad = True
    for name, param in model.named_parameters():
         if param.requires_grad and name in freeze_layers_names:
             param.requires_grad = False
    print('[INFO]: Freezing %d percent of the top trainable layers...' % (100-args.ratio_finetune))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CyclicLR(optimizer=optimizer, base_lr=args.max_lr/100, max_lr=args.max_lr,mode='triangular2',cycle_momentum=False)
    model.to(device)    
    epoch_num = args.num_epochs
    best_val_acc = 0
    best_val_bacc=0
    total_loss_val, total_acc_val = [],[]
    for epoch in tqdm(range(1, epoch_num+1)):
        criterion_hm=None
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).to(device)
        elif args.loss_type == 'IB':
            criterion = nn.CrossEntropyLoss(weight=None).to(device)
            criterion_hm = IBLoss(weight=None).to(device)
        elif args.loss_type == 'Focal':
            criterion = FocalLoss(weight=None).to(device)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(weight=None,cls_num_list=cls_num_list).to(device)
        elif args.loss_type == 'SHML1':
                criterion = SoftmaxHardMiningLoss1(weight=per_cls_weights,delta=args.delta).to(device)
        elif args.loss_type == 'SHML2':
                criterion = SoftmaxHardMiningLoss2(weight=per_cls_weights).to(device)
        elif args.loss_type == 'CHML1':
                criterion = nn.CrossEntropyLoss(weight=None).to(device)
                criterion_hm = SoftmaxHardMiningLoss1(weight=per_cls_weights,delta=args.delta).to(device)
        elif args.loss_type == 'CHML2':
             criterion =  criterion = nn.CrossEntropyLoss(weight=None).to(device)
             criterion_hm =  SoftmaxHardMiningLoss2(weight=per_cls_weights).to(device)
        elif args.loss_type == 'CHML3':
                criterion = SoftmaxHardMiningLoss2(weight=None).to(device)
                criterion_hm = SoftmaxHardMiningLoss1(weight=per_cls_weights,delta=args.delta).to(device)
        elif args.loss_type == 'CHML4':
             criterion =  SoftmaxHardMiningLoss2(weight=None).to(device)
             criterion_hm =  SoftmaxHardMiningLoss2(weight=per_cls_weights).to(device)
        elif args.loss_type == 'DHML1':
                criterion = SoftmaxHardMiningLoss2(weight=None).to(device)
                criterion_hm = DynamicalMiningLoss1(weight=per_cls_weights,delta=args.delta,epoch=args.actual_epoch,max_thresh=args.max_thresh,numb_epochs=args.num_epochs).to(device)
        elif args.loss_type == 'DHML2':
                criterion = SoftmaxHardMiningLoss2(weight=None).to(device)
                criterion_hm = DynamicalMiningLoss2(weight=per_cls_weights,epoch=args.actual_epoch,max_thresh=args.max_thresh,numb_epochs=args.num_epochs).to(device)

        else:
            warnings.warn('Loss type is not listed')
        loss_train, acc_train, total_loss_train, total_acc_train = train(train_loader, model, criterion, criterion_hm,optimizer,args, epoch, device)
        loss_val, acc_val = validate(val_loader, model, criterion, criterion_hm, optimizer,args, epoch, device)
        val_bacc, val_auc, y_val,y_pred_val= backup(val_loader, model,  epoch, device)
        scheduler.step()
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)
        if args.distribution and run=='run0' :
            if epoch==1:
                def_pred_distr_history=pd.DataFrame(columns=['Epoch','0.0-0.05','0.05-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1','Total mel'])
                nev_pred_distr_history=pd.DataFrame(columns=['Epoch','0.0-0.05','0.05-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1','Total nev'])
                proba_pred_distr_history=pd.DataFrame(columns=['Epoch','0.0-0.05','0.05-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1','Total images'])
            def_dist_row,nev_dist_row,proba_dist_row=eval_distribution(y_val,y_pred_val,epoch)
            def_pred_distr_history=pd.concat([def_pred_distr_history,def_dist_row])
            nev_pred_distr_history=pd.concat([nev_pred_distr_history,nev_dist_row])
            proba_pred_distr_history=pd.concat([proba_pred_distr_history,proba_dist_row])
            def_pred_distr_history.to_csv(args.history_path+args.store_name+'/def_proba_distribtuion_history.csv')
            nev_pred_distr_history.to_csv(args.history_path+args.store_name+'/nev_proba_distribtuion_history.csv')
            proba_pred_distr_history.to_csv(args.history_path+args.store_name+'/proba_distribtuion_history.csv')
        if val_bacc>=best_val_bacc:
            print('------------------------------------------------------------')
            print('Increase of best bacc on [epoch %d] from [%.5f] to [%.5f]' % (epoch, best_val_bacc,  val_bacc))
            print('------------------------------------------------------------')
            best_val_bacc=val_bacc
            best_epoch=epoch
            patience=args.delay
            torch.save(model.state_dict(), args.model_path+args.store_name+'/'+f'best_model_{run}.pth')  # Saving current best model
            print('------------------------------------------------------------')
            print('Saving current best model')
            print('------------------------------------------------------------')
        else:
            patience -= 1
            if 'DHML' in args.loss_type and patience<=args.delay_hm:
                args.hml=True
                args.actual_epoch=epoch
                patience=args.delay
                print('------------------------------------------------------------')
                print('No improvement of best bacc on [epoch %d] since epoch [%d]' % (epoch, epoch-(args.delay-args.delay_hm)))
                print('------------------------------------------------------------')
                print('------------------------------------------------------------')
                print('training with DMLoss on [epoch %d]' % (epoch))
                print('------------------------------------------------------------')
            elif patience == 0:
                print('Early stopping on [epoch: %d], [Best bacc Val: %.3f'%(epoch,best_val_bacc))
                break

    learning_curve(total_acc_train,total_acc_val,total_loss_train,total_loss_val,args.history_path+args.store_name)
    print ('Evaluating the model')
    model.load_state_dict(torch.load(args.model_path+args.store_name+'/'+f'best_model_{run}.pth'))
    print('Restore best model on epoch %d with Best bacc Val: %.3f'%(best_epoch,best_val_bacc))
    model.eval()
    y_true = []
    y_pred = []
    y_score=[]
    with torch.no_grad():
        for _, data in enumerate(test_loader):
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
        test_bacc = balanced_accuracy_score(y_true, y_pred)
        test_auc=roc_auc_score(y_true_2D, y_score)
    current_result_save=pd.DataFrame(np.array([[run,best_epoch,test_bacc,test_auc]]),columns=['run','best_epoch','bacc','auc'])
    result_save=pd.concat([result_save,current_result_save])
    result_save.to_csv(args.result_dir+args.store_name+'/test_history2.csv')
    print('------------------------------------------------------------')
    print('[test bacc %.5f] ,  [test auc %.5f]' % (test_bacc,test_auc))
    print('------------------------------------------------------------')
    plot_labels = ['ACK','BCC','BEK','DEF','MEL','NEV','VAL']
    report = classification_report(y_true, y_pred, target_names=plot_labels)
    print(report)

    
