# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:24:59 2022

@author: arthu
"""

import pandas as pd
case='bhml'
data=pd.read_csv("proba_pred_distribution_bhml_run0.csv")
data['prob']=0


for i in range(len(data)):
    if data.Probabilities[i]=='[0.0-0.05]':
        data.prob[i]=1
    elif data.Probabilities[i]=='[0.05-0.1]':
        data.prob[i]=2
    elif data.Probabilities[i]=='[0.1-0.2]':
        data.prob[i]=3
    elif data.Probabilities[i]=='[0.2-0.3]':
        data.prob[i]=4
    elif data.Probabilities[i]=='[0.3-0.4]':
        data.prob[i]=5
    elif data.Probabilities[i]=='[0.4-0.5]':
        data.prob[i]=6
    elif data.Probabilities[i]=='[0.5-0.6]':
        data.prob[i]=7
    elif data.Probabilities[i]=='[0.6-0.7]':
        data.prob[i]=8
    elif data.Probabilities[i]=='[0.7-0.8]':
        data.prob[i]=9
    elif data.Probabilities[i]=='[0.8-0.9]':
        data.prob[i]=10
    elif data.Probabilities[i]=='[0.9-1]':
        data.prob[i]=11

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
X, Y, Z = axes3d.get_test_data(0.05)

# Plot a basic wireframe.
ax.scatter(data.prob, data.Epoch, data.Number_of_samples)
ax.set_title('Proba distribution of per epoch of all classes with '+case+' loss',fontsize=20)
ax.set_xlabel('Probabilities range')
ax.set_ylabel('Epoch')
ax.set_zlabel('Number of samples')

plt.show()
