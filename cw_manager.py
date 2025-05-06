# -*- coding: utf-8 -*-
"""
Created on Wed May  7 00:41:32 2025

@author: 18307
"""

import os
import pandas as pd

def read_channel_weight_connectivity_matrix():
    

def read_channel_weight_target(identifier='label_driven_mi', sort=False):
    path_current = os.getcwd()
    path_file = os.path.join(path_current, 'channel_weights', 'channel_weights_target.xlsx')
    
    channel_weight = pd.read_excel(path_file, sheet_name=identifier, engine='openpyxl')
    weight = channel_weight[['labels','ams']]
    
    if sort:
        weight = weight.sort_values(by='ams', ascending=False)
    
    return weight

def read_channel_weight_fitting(model_fm='basic', model_rcm='differ', model='exponential', sort=False):
    model_fm = model_fm.lower()
    model_rcm = model_rcm.lower()
    model = model.lower()
    
    if model_fm == 'basic':
        token_model_fm = 'BFM'
    elif model_fm == 'advanced':
        token_model_fm = 'AFM'
    
    if model_rcm == 'differ':
        token_model_rcm = 'DRCM'
    elif model_rcm == 'linear':
        token_model_rcm = 'LRCM'
    elif model_rcm == 'linear_ratio':
        token_model_rcm = 'LRRCM'
    
    path_current = os.getcwd()
    path_file = os.path.join(path_current, 'channel_weights', 
                             f'channel_weights_{token_model_fm}_{token_model_rcm}_LDMITG.xlsx')
    
    channel_weight = pd.read_excel(path_file, sheet_name=model, engine='openpyxl')
    
    weight = channel_weight[['labels','ams']]
    
    if sort:
        weight = weight.sort_values(by='ams', ascending=False)
    
    return weight

if __name__ == '__main__':
    weight_target = read_channel_weight_target(identifier='label_driven_mi', sort=True)
    weight_fitting = read_channel_weight_fitting(model_fm='basic', model_rcm='differ', model='exponential', sort=False)
    