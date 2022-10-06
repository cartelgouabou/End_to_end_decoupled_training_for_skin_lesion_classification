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


df_train, df_test, df_val, cls_num_list = get_data(root_path+args.dataset_dir)

df_train.to_csv(root_path+'/base/df_train.csv')
df_test.to_csv(root_path+'/base/df_test.csv')
df_val.to_csv(root_path+'/base/df_val.csv')
cls_num_list_pd=pd.DataFrame(cls_num_list)
cls_num_list_pd.to_csv(root_path+'/base/cls_num_list_pd.csv')

