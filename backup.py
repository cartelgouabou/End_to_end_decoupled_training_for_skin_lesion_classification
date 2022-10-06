# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:35:17 2021

@author: arthu
"""


from __future__ import print_function, division
import tensorflow as tf
from numpy.random import seed
seed(12)

import numpy as np
import pandas as pd
#from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from math import exp
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,f1_score
from time import time
from tensorflow.keras.utils import to_categorical
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()
        self.step_size = step_size
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
        


class BaccCallback(Callback):
    """This callback implements a customized early stopping.
    The method stop the tuning if the validation BACC(Balanced accuracy) 
    failed to improve for a certain number of epoch called delay
    it has been implement as detailed in this paper (https:......).
    it doesn't allow generator
    It also save history of the training on a specified directory in .txt format
    For more detail, please see paper.
    
    # Example
        ```python
            bcb = BaccCallback(X_valid,Y_valid,step,path_history,path_weights,task_name,case_loss_fn,split,delay,zero_one_label_range)
            model.fit(X_train, Y_train, callbacks=[bcb])
        ```

    # Arguments
        X_valid: initial learning rate which is the
            lower boundary in the cycle.
        Y_valid: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        path_history: path where to save training history
            If scale_fn is not None, this argument is ignored.
        path_weights: path where to save best weights
        task_name: specified task name in string
        case_loss_fn: specified case name in string.
        split: specified split name
        delay: number of epoch to waiting before stopping training 
        zero_one_label_range: boolean function specified the label range [0,1] or [-1,1]
    """
    def __init__(self,x_valid,y_valid,path_history,path_weights,task,case,split,delay,epochs):
        self.x_val = x_valid
        self.y_val = y_valid
        self.path_history = path_history
        self.path_weights = path_weights
        self.best_bacc_val = 0
        self.best_roc_val = 0
        self.best_acc_val = 0# np.Inf
        self.best_acc = 0# np.Inf
        self.best_val_loss = 0# np.Inf
        self.best_loss = 0# np.Inf
        self.best_epoch = 0
        self.best_f1_val=0
        self.task=task
        self.case=case
        self.split=split
        self.wait = 0
        self.delay = delay  #16 epochs
        self.best_training_time =0
        self.start_time=0
        self.ep=epochs
        
    def on_epoch_begin(self,epoch, logs=None):
        if (epoch==0) :
            self.start_time=time()
            print('START task : {}'.format(self.task))
            f=open(self.path_history+'train_history_{}_'.format(self.task)+'_{}'.format(self.case)+'_{}_'.format(self.split) +'.txt',"a+")
            line=['START OF task : {}\r\n'.format(self.task),"Case : {}\r\n" .format(self.case)]
            f.writelines(line)
            f.close()
    def on_epoch_end(self, epoch, logs={}):
        #evaluate delay before earling stopping
        #evaluate performance on validation set
        y_pred_val = self.model.predict(self.x_val)
        y_pred_val2 = np.array([1 if p>=0.5 else 0 for p in y_pred_val])
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        bacc_val= balanced_accuracy_score(self.y_val, y_pred_val2)
        val_acc=logs.get('val_accuracy')
        acc=logs.get('accuracy')
        loss=logs.get('loss')
        val_loss=logs.get('val_loss')
        f1_val=f1_score(self.y_val, y_pred_val2)
        
        if  (np.less(self.best_bacc_val,bacc_val)) :
            self.wait=0
            self.best_acc=acc
            self.best_acc_val = val_acc
            self.best_roc_val = roc_val
            self.best_bacc_val = bacc_val
            self.best_epoch = epoch
            self.best_f1_val=f1_val
            self.best_val_loss = val_loss
            self.best_loss = loss
            self.best_training_time = time()-self.start_time
            self.model.save_weights(self.path_weights+'best_weights_'+self.task+'_'+self.case+'_'+self.split+'.hdf5')
            callbacks_save=pd.DataFrame(np.array([[self.best_loss,self.best_val_loss,self.best_training_time,self.best_bacc_val]]),columns=['loss','loss_val','training_time','best_bacc_val'])
            callbacks_save.to_csv(self.path_history+'best_callbacks_'+self.task+'_'+self.case+'_'+self.split+'.csv')
            
            if (epoch == (self.ep-1)):
               print('\model training stopped at epoch{}'.format(epoch)) 
               f=open(self.path_history+'train_history_{}_'.format(self.task)+'_{}'.format(self.case)+'_{}_'.format(self.split)+'.txt',"a+")
               line=['END OF training\r\n',"num of epoch reach :%s\r\n" %(epoch),"Acc:%s\r\n" %(acc*100),"Val_acc:%s\r\n" %(val_acc*100),"Current_bacc_val:%s\r\n" %(bacc_val*100),"Best_bacc_val:%s\r\n" %(self.best_bacc_val*100)]
               lines=['BEST ACHIEVE during training :\r\n',"Best_Epoch:%s\r\n" %(self.best_epoch),"Best_loss:%s\r\n" %(self.best_loss),"Best_val_loss:%s\r\n" %(self.best_val_loss),"Best_acc:%s\r\n" %(self.best_acc*100),"Best_acc_val:%s\r\n" %(self.best_acc_val*100),"Best_roc_val:%s\r\n" %(self.best_roc_val*100),"Best_bacc_val:%s\r\n" %(self.best_bacc_val*100),"Best_f1_val:%s\r\n" %(self.best_f1_val*100),"Best_training_time:%s\r\n" %(self.best_training_time)]
               f.writelines(line)
               f.writelines(lines)
               f.close()
        else:
            self.wait+=1
            if self.wait >= self.delay: 
                self.model.stop_training = True
                print('1\Early model training stopped due to not improvment of BACC_val at epoch{}'.format(epoch))
                f=open(self.path_history+'train_history_{}_'.format(self.task)+'_{}'.format(self.case)+'_{}_'.format(self.split) +'.txt',"a+")
                line=['END OF training due to not improvment of BACC_val\r\n','Early_stop\r\n',"Epoch:%s\r\n" %(epoch),"Acc:%s\r\n" %(acc*100),"Val_acc:%s\r\n" %(val_acc*100),"Current_bacc_val:%s\r\n" %(bacc_val*100),"Best_bacc_val:%s\r\n" %(self.best_bacc_val*100)]
                lines=['BEST ACHIEVE during training :\r\n',"Best_Epoch:%s\r\n" %(self.best_epoch),"Best_loss:%s\r\n" %(self.best_loss),"Best_val_loss:%s\r\n" %(self.best_val_loss),"Best_acc:%s\r\n" %(self.best_acc*100),"Best_acc_val:%s\r\n" %(self.best_acc_val*100),"Best_roc_val:%s\r\n" %(self.best_roc_val*100),"Best_bacc_val:%s\r\n" %(self.best_bacc_val*100),"Best_f1_val:%s\r\n" %(self.best_f1_val*100),"Best_training_time:%s\r\n" %(self.best_training_time)]
                f.writelines(line)
                f.writelines(lines)
                f.close()
            elif (epoch == (self.ep-1)):
               print('\model training stopped at epoch{}'.format(epoch)) 
               f=open(self.path_history+'train_history_{}_'.format(self.task)+'_{}'.format(self.case)+'_{}_'.format(self.split)+'.txt',"a+")
               line=['END OF training\r\n',"num of epoch reach :%s\r\n" %(epoch),"Acc:%s\r\n" %(acc*100),"Val_acc:%s\r\n" %(val_acc*100),"Current_bacc_val:%s\r\n" %(bacc_val*100),"Best_bacc_val:%s\r\n" %(self.best_bacc_val*100)]
               lines=['BEST ACHIEVE during training :\r\n',"Best_Epoch:%s\r\n" %(self.best_epoch),"Best_loss:%s\r\n" %(self.best_loss),"Best_val_loss:%s\r\n" %(self.best_val_loss),"Best_acc:%s\r\n" %(self.best_acc*100),"Best_acc_val:%s\r\n" %(self.best_acc_val*100),"Best_roc_val:%s\r\n" %(self.best_roc_val*100),"Best_bacc_val:%s\r\n" %(self.best_bacc_val*100),"Best_f1_val:%s\r\n" %(self.best_f1_val*100),"Best_training_time:%s\r\n" %(self.best_training_time)]
               f.writelines(line)
               f.writelines(lines)
               f.close()
        return
    
    
    
        
