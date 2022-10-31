#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:05:50 2022

@author: arthur
"""
import argparse
import os
import pandas as pd
import numpy as np
root_path=os.getcwd()
parser = argparse.ArgumentParser(description='Cifar Generate results')
parser.add_argument('--dataset', default='cifar10', help='dataset setting',dest='dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32')
parser.add_argument('--loss_type', default="SHML1", type=str, help='loss type')
parser.add_argument('--weighting_type', default='CS', type=str, help='data sampling strategy for train loader')
parser.add_argument('--imb_type', default="both", type=str, help='imbalance type')
parser.add_argument('--num_runs', default=10, type=int, help='number of runs to launch ') 
parser.add_argument('--imb_ratio_list','--ir_list', nargs='+',metavar='N', type=float, default=0.02, help='imbalance ratio, 1 for balanced, 0.1 or 0.01; use case: --ir 1 0.1 0.01 0.02',dest='ir_list')
parser.add_argument('--result_dir', default='/results/', type=str, help='results path ')
parser.add_argument('--ratio_finetune', default=32, type=float, help='ratio of last layer to finetune')
parser.add_argument('--distribution', type=bool, default=True, help='Visualize data distribution')
parser.add_argument('--delta', default=10000000, type=int, help='delta parameter for HMLoss')
parser.add_argument('--hm_delay_type', default='epoch', type=str, help='data sampling strategy for train loader')
parser.add_argument('--max_thresh', default=0.1, type=float,   help='max thresh for outliers')
args = parser.parse_args()


path_history=root_path+'/history/'
path_stat=root_path+'/stats/'
if not os.path.exists(path_stat):
      os.makedirs(path_stat)

metric_list=['mean_BACC','std_BACC','mean_AUC','std_AUC','median_BACC','median_AUC','mean_best_epoch','std_mean_b_epoch']


data=[]
if 'DHML' in args.loss_type:
    args.store_name = '_'.join(['isic2018', 'efficientb3', args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune),'D',str(args.delta)])
else:
    args.store_name = '_'.join(['isic2018', 'efficientb3', args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)])

       
#statistic=pd.read_csv(root_path+args.result_dir+args.store_name+'/test_history.csv')
statistic=pd.read_csv(root_path+args.result_dir+args.store_name+'/test_history2.csv')
#statistic=pd.read_csv(path_statistic)
stats=[np.mean([statistic.bacc[0],statistic.bacc[1],statistic.bacc[2],statistic.bacc[3],statistic.bacc[4],statistic.bacc[0],statistic.bacc[1],statistic.bacc[2],statistic.bacc[3],statistic.bacc[4]]), #ACC
        np.std([statistic.bacc[0],statistic.bacc[1],statistic.bacc[2],statistic.bacc[3],statistic.bacc[4],statistic.bacc[0],statistic.bacc[1],statistic.bacc[2],statistic.bacc[3],statistic.bacc[4]]),#Err
        np.mean([statistic.auc[0],statistic.auc[1],statistic.auc[2],statistic.auc[3],statistic.auc[4],statistic.auc[0],statistic.auc[1],statistic.auc[2],statistic.auc[3],statistic.auc[4]]), #ACC
        np.std([statistic.auc[0],statistic.auc[1],statistic.auc[2],statistic.auc[3],statistic.auc[4],statistic.auc[0],statistic.auc[1],statistic.auc[2],statistic.auc[3],statistic.auc[4]]),
        np.median([statistic.bacc[0],statistic.bacc[1],statistic.bacc[2],statistic.bacc[3],statistic.bacc[4],statistic.bacc[0],statistic.bacc[1],statistic.bacc[2],statistic.bacc[3],statistic.bacc[4]]),
        np.median([statistic.auc[0],statistic.auc[1],statistic.auc[2],statistic.auc[3],statistic.auc[4],statistic.auc[0],statistic.auc[1],statistic.auc[2],statistic.auc[3],statistic.auc[4]]),
        np.mean([statistic.best_epoch[0],statistic.best_epoch[1],statistic.best_epoch[2],statistic.best_epoch[3],statistic.best_epoch[4],statistic.best_epoch[0],statistic.best_epoch[1],statistic.best_epoch[2],statistic.best_epoch[3],statistic.best_epoch[4]]),
        np.std([statistic.best_epoch[0],statistic.best_epoch[1],statistic.best_epoch[2],statistic.best_epoch[3],statistic.best_epoch[4],statistic.best_epoch[0],statistic.best_epoch[1],statistic.best_epoch[2],statistic.best_epoch[3],statistic.best_epoch[4]])
     ]       
data=np.array([stats])
p_stat=pd.DataFrame(data,
                #index=case,
                columns=metric_list)
if 'DHML' in args.loss_type:
    name_result='_'.join(['statistic','efficientb3',args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune),'D',str(args.delta),'.csv'])
else:
    args.store_name = '_'.join(['statistic','isic2018', 'efficientb3', args.loss_type, 'W' ,args.weighting_type,'F',str(args.ratio_finetune)+'.csv'])
p_stat.to_csv(path_stat+name_result)
        
            

