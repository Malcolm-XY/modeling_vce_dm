# -*- coding: utf-8 -*-
"""
Created on Wed May  7 00:41:32 2025

@author: 18307
"""

import os
import pandas as pd

def read_channel_importances_DD(identifier='data_driven_pcc', sort=False):
    path_current = os.getcwd()
    path_file = os.path.join(path_current, 'channel_importances', 'channel_importances_DD.xlsx')
    
    channel_importances = pd.read_excel(path_file, sheet_name=identifier, engine='openpyxl')
    importances = channel_importances[['labels','ams']]
    
    if sort:
        importances = importances.sort_values(by='ams', ascending=False)
    
    return importances
    
def read_channel_importances_LD(identifier='label_driven_mi', sort=False):
    path_current = os.getcwd()
    path_file = os.path.join(path_current, 'channel_importances', 'channel_importances_LD.xlsx')
    
    channel_importances = pd.read_excel(path_file, sheet_name=identifier, engine='openpyxl')
    importances = channel_importances[['labels','ams']]
    
    if sort:
        importances = importances.sort_values(by='ams', ascending=False)
    
    return importances

def read_channel_importances_fitted(model_fm='basic', model_rcm='differ', model='exponential', 
                                source='fitted_results(15_15_joint_band_from_mat)', sort=False):
    model_fm = model_fm.lower()
    model_rcm = model_rcm.lower()
    model = model.lower()
    
    path_current = os.getcwd()
    path_file = os.path.join(path_current, 'fitted_results', source, 
                             f'channel_weights({model_fm}_fm_{model_rcm}_rcm).xlsx')

    channel_importances = pd.read_excel(path_file, sheet_name=model, engine='openpyxl')
    
    importances = channel_importances[['labels','ams']]
    
    if sort:
        importances = importances.sort_values(by='ams', ascending=False)
    
    return importances

if __name__ == '__main__':
    # %% For 15/15 subjects; 3/3 data in dataset
    
    # weight_control = read_channel_weight_DD(identifier='data_driven_pcc', sort=True)
    # weight_target = read_channel_weight_LD(identifier='label_driven_mi', sort=True)   
     # weight_fitting = read_channel_weight_fitting(model_fm='basic', model_rcm='differ', model='exponential', sort=True)
     
    # %% For 10/15 subjects; 2/3 data in dataset
    importances_control = read_channel_importances_DD(identifier='data_driven_pcc_10_15', sort=True)
    importances_target = read_channel_importances_LD(identifier='label_driven_mi_1_5', sort=True)
    importances_fitted = read_channel_importances_fitted(model_fm='basic', model_rcm='differ', model='exponential',
                                                 source='fitted_results(10_15_joint_band_from_mat)', sort=True)