import os
import pandas as pd
import numpy as np
root_path=os.getcwd()
case='isic2018_efficientb3_CE_W_None_F_32'
path_checkpoint =root_path+'/checkpoint/'+case+'/'
filenames_def='def_proba_distribtuion_history.csv'
filenames_out='def_pred_distribution_'+case+'.csv'
data=pd.read_csv(path_checkpoint+filenames_def)
data_reshaped=pd.DataFrame(columns=['Probabilities','Epoch','Number_of_samples'])
Proba_list=['0.0-0.05','0.05-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1']
for prob in Proba_list:
    for epoch in range(len(data)):
        new_row=pd.DataFrame(np.array([['['+prob+']',epoch,data[prob][epoch]]]),
                                                         columns=['Probabilities','Epoch','Number_of_samples'])
        data_reshaped=pd.concat([data_reshaped,new_row])
data_reshaped.to_csv(filenames_out)

filenames_nev='nev_proba_distribtuion_history.csv'
filenames_out='nev_pred_distribution_'+case+'.csv'
data=pd.read_csv(path_checkpoint+filenames_nev)
data_reshaped=pd.DataFrame(columns=['Probabilities','Epoch','Number_of_samples'])
Proba_list=['0.0-0.05','0.05-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1']
for prob in Proba_list:
    for epoch in range(len(data)):
        new_row=pd.DataFrame(np.array([['['+prob+']',epoch,data[prob][epoch]]]),
                                                         columns=['Probabilities','Epoch','Number_of_samples'])
        data_reshaped=pd.concat([data_reshaped,new_row])
data_reshaped.to_csv(filenames_out)

filenames_pro='proba_distribtuion_history.csv'
filenames_out='proba_pred_distribution_'+case+'.csv'
data=pd.read_csv(path_checkpoint+filenames_pro)
data_reshaped=pd.DataFrame(columns=['Probabilities','Epoch','Number_of_samples'])
Proba_list=['0.0-0.05','0.05-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1']
for prob in Proba_list:
    for epoch in range(len(data)):
        new_row=pd.DataFrame(np.array([['['+prob+']',epoch,data[prob][epoch]]]),
                                                         columns=['Probabilities','Epoch','Number_of_samples'])
        data_reshaped=pd.concat([data_reshaped,new_row])
data_reshaped.to_csv(filenames_out)