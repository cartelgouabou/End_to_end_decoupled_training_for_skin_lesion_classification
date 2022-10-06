#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:30:55 2022

@author: arthur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:22:14 2022

@author: arthur
"""

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from data_utils import get_data,CustomImageDataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import argparse
import os
import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
root_path=os.getcwd()
parser = argparse.ArgumentParser(description='PyTorch t-SNE for isic2018')
parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')
parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, help='random seed value (default: 1)')
parser.add_argument('--dataset_dir', default='/dataset/', help='dataset setting')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)
    
input_size = 300
# set dataset
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.59451115, 0.59429777, 0.59418184],
                                                            [0.14208533, 0.18548788, 0.20363748])])

# if args.dataset == 'cifar10':
#     val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
#     num_classes = 10
# elif args.dataset == 'cifar100':
#     val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
#     num_classes = 100
# dataloader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=100, shuffle=False,
#     num_workers=args.workers, pin_memory=True)

df_train, df_test, df_val, cls_num_list = get_data(root_path+args.dataset_dir)
validation_set = CustomImageDataset(df_val, transform=val_transform)
val_loader = DataLoader(validation_set, batch_size=args.batch_size//2, shuffle=False, num_workers=args.workers)

# set model
# use_norm =  False
# net = modelss.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)
# net = net.to(device)
# net.load_state_dict(torch.load('/media/arthur/Data/PROJET/Class_imbalance/TSNE/checkpoint/best_model.pth'))
# #net = models.resnet50(pretrained=True)

net = models.efficientnet_b3(pretrained=True)
net.classifier[1]=nn.Linear(1536, 7)

net=torch.load(root_path+'/model/efficientnetb3_model.pth')
net.load_state_dict(torch.load('/media/arthur/Data/PROJET/Class_imbalance/github/ACCV_github/TSNE/checkpoint/isic2019_efficientb3_CE_W_None_F_32_Dis_True_EPOCH_50/best_model_run0.pth'))

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

def gen_features():
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_np = targets.data.cpu().numpy()

            outputs = net(inputs)
            outputs_np = outputs.data.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(val_loader)):
                print(idx+1, '/', len(val_loader))

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, outputs

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 7),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,'tsne_CE_50_2.png'), bbox_inches='tight')
    print('done!')

targets, outputs = gen_features()
tsne_plot(args.save_dir, targets, outputs)