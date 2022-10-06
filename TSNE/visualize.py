#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 00:19:35 2022

@author: arthur
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
def learning_curve(total_acc_train,total_acc_val,total_loss_train,total_loss_val,history_path):
    cols = ['{}'.format(col) for col in ['Model accuracy ','Model loss ']]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 20))
    for ax, col in zip(axes, cols):
            ax.set_title(col,fontsize=20)
    # Historique des précisions
    axes[0].plot(total_acc_train,label='training') 
    axes[0].plot(total_acc_val,label='validation')
    axes[0].legend(loc="upper right",fontsize=20)
    axes[0].set_xlabel('epoch',fontsize=20)
    axes[0].set_ylabel('accuracy',fontsize=20)
    
    # Historique des erreurs
    axes[1].plot(total_loss_train,label='training')   
    axes[1].plot(total_loss_val,label='validation')
    axes[1].legend(loc="upper right",fontsize=20)
    axes[1].set_xlabel('epoch',fontsize=20)
    axes[1].set_ylabel('loss',fontsize=20)
    plt.savefig(history_path+'/history_case_plot.png')
    fig.tight_layout()
    plt.close()
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def view_samples(imageid_path_dict, all_image_path):

    fig = plt.figure(figsize=(15, 15))
    columns, rows = 3, 2
    start, end = 0, len(imageid_path_dict)
    ax = []
    import random
    for i in range(columns*rows):
        k = random.randint(start, end)
        img = mpimg.imread((all_image_path[k]))
        ax.append( fig.add_subplot(rows, columns, i+1) )
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap="gray")
    plt.tight_layout(True)
    plt.show()  # finally, render the plot