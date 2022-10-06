#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:09:32 2022

@author: arthur
"""

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


for i in range(len(df_test)):
    df_test.path[i]=root_path+df_test.path[i][46:]

args.weighting_type='CS'
args.loss_type='CHML3'

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
run_list=['run6']
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
        
    print ('Evaluating the model')
    model.load_state_dict(torch.load(args.model_path+args.store_name+'/'+f'best_model_{run}.pth'))
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
    print('------------------------------------------------------------')
    print('[test bacc %.5f] ,  [test auc %.5f]' % (test_bacc,test_auc))
    print('------------------------------------------------------------')
    plot_labels = ['ACK','BCC','BEK','DEF','MEL','NEV','VAL']
    report = classification_report(y_true, y_pred, target_names=plot_labels)
    print(report)

    


from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, plot_confusion_matrix,f1_score

cm = confusion_matrix(y_true, y_pred)
cm = np.array(confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5,6]))
confusion = pd.DataFrame(cm, index=['ACK','BCC','BEK','DEF','MEL','NEV','VAL'],
                         columns=['predict_ACK','predict_BCC','predict_BEK','predict_DEF','predict_MEL','predict_NEV','predict_VAL'])
confusion


target_names = ['ACK','BCC','BEK','DEF','MEL','NEV','VAL']
print(classification_report(y_true, y_pred, target_names=target_names))

print('Balanced_accuracy: %.3f' % balanced_accuracy_score(y_true,y_pred))

#Tracer de la courbe ROC

scores= y_score    #prédit le score de confiance des prédiction,il s'agit de la distance entre le vecteur et l'hyperplan
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve
from matplotlib import pyplot

# Calcul de l'AUC
auc = roc_auc_score(y_true_2D, scores,average='weighted')
auc = roc_auc_score(y_true_2D[:,0], y_score[:,0],average='weighted')
print('AUC: %.3f' % auc)
# Evaluation de la courbe ROC
fpr = dict()
tpr=dict()
thd=dict()
roc_auc=dict()
for i in range(7):      
    fpr[i], tpr[i], thd[i] = roc_curve(y_true_2D[:,i], y_score[:,i])
    roc_auc[i]=roc_auc_score(y_true_2D[:,i], y_score[:,i],average='weighted')
    (fpr[i], tpr[i])
from itertools import cycle    
colors=cycle(['brown','red','green','purple','blue','black','orange']) #,'red','purple'
target_names = ['ACK','BCC','BEK','DEF','MEL','NEV','VAL']
for i, color in zip(range(7),colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='{0} (AUC = {1:0.2f})'
             ''.format(target_names[i], roc_auc[i]))
# #moin class
# for i in [1,2]:
#     fpr[i], tpr[i], thd[i] = roc_curve(y_test_2D[:,i], scores[:,i])
#     roc_auc[i]=roc_auc_score(y_test_2D[:,i], scores[:,i])
#     (fpr[i], tpr[i])
# from itertools import cycle    
# colors=cycle(['brown','red','green']) #,'red','purple'
# target_names = ['BEK', 'MEL','NEV']
# for i, color in zip([1,2],colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='ROC curve of class {0} (AUC = {1:0.2f})'
#              ''.format(target_names[i], roc_auc[i]))
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# Traver la courbe ROC du modèle 
#plt.plot(fpr, tpr, marker='.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('1-specificity')
plt.ylabel('Sensitivity')
plt.title('ROC curve')
plt.legend(loc="lower right")
# Afficher la courbe
plt.savefig('auc_plot_fr.png')
plt.show()

# Calcul de l'AUPRC

# # Evaluation de la courbe PRC
# prec = dict()
# reca=dict()
# #thd=dict()
# prc_auc=dict()
# for i in range(7):
#     prec[i], reca[i], _ = precision_recall_curve(y_true_2D[:,i], y_score[:,i])
#     prc_auc[i]=auc(reca[i], prec[i])
#     (prec[i], reca[i])
# from itertools import cycle    
# colors=cycle(['brown','red','green','purple','blue','black','orange']) #,'red','purple'
# target_names = ['ACK','BCC','BEK','DEF','MEL','NEV','VAL']
# for i, color in zip(range(7),colors):
#     plt.plot(reca[i], prec[i], color=color, lw=2,
#              label='PR curve of class {0} (AUC = {1:0.2f})'
#              ''.format(target_names[i], prc_auc[i]))

# # plot no skill
# plt.plot([0, 1], [0, 1], linestyle='--')
# # Traver la courbe PRC du modèle 
# #plt.plot(fpr, tpr, marker='.')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# pyplot.xlabel('Recall')
# pyplot.ylabel('Precision')
# pyplot.title('Precision Recall Curve')
# plt.legend(loc="lower right")
# # Afficher la courbe
# plt.savefig('prc_plot.png')
# plt.show()