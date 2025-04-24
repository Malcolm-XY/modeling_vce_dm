# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:11:18 2025

@author: 18307
"""

import os
import pickle
import numpy as np
import pandas as pd

import utils_ as utils
from utils import utils_feature_loading


def read_pkl(path_file, method='pd'):
    if method == 'pd':
        data = pd.read_pickle(path_file)
    
    elif method == 'pkl':
        with open(path_file, 'rb') as file:
            data = pickle.load(file)
        
    return data

def read_functional_connectivity(identifier, feature, method='pkl', dtype='np'):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_parent_parent = os.path.dirname(path_parent)
    path_fc_features = os.path.join(path_parent_parent, 'Research_Data', 'SEED', 'functional connectivity')
    
    if method.lower() == 'pkl':
        path_data = os.path.join(path_fc_features, f'{feature.lower()}_pkl', f'{identifier.lower()}.pkl')
        data = utils.read_pkl(path_data)
    
    if dtype.lower() == 'np':
        data = np.array(data)
    
    return data

def compute_mean_functional_connectivity(feature, subject_range=range(1,16), experiment_range=range(1,4), band='joint'):
    data = []
    for subject in subject_range:
        for experiment in experiment_range:
            if band.lower() == 'joint':
                alpha = read_functional_connectivity(identifier=f'sub{subject}ex{experiment}_alpha', feature=feature)
                beta = read_functional_connectivity(identifier=f'sub{subject}ex{experiment}_beta', feature=feature)
                gamma = read_functional_connectivity(identifier=f'sub{subject}ex{experiment}_gamma', feature=feature)
                
                data_temp = (alpha + beta + gamma)/3
                
            elif band.lower() in ['alpha', 'beta', 'gamma']:
                data_temp = read_functional_connectivity(identifier=f'sub{subject}ex{experiment}_{band.lower()}', feature=feature)
            
            data.append(data_temp)
    
    data = np.array(data)
    data_mean = data.mean(axis = (0, 1, 2))
    
    return data_mean

# %% Example Usage
if __name__ == '__main__':
    # # get electrodes
    # distribution = utils.get_distribution()
    # electrodes = distribution['channel']
    
    # # compute mis_mean
    # mis_mean = compute_mean_functional_connectivity('mi', subject_range=range(1,2), experiment_range=range(1,4))
    
    # # arrange mis_mean
    # mis_mean_ = pd.DataFrame({'electrodes':electrodes, 'mi_mean':mis_mean})
    
    # # plot heatmap
    # utils.draw_heatmap_1d(mis_mean, electrodes)
    # utils.draw_heatmap_1d(np.log(mis_mean), electrodes)
    
    # # get ascending indices
    # mis_mean_resorted = mis_mean_.sort_values('mi_mean', ascending=False)
    # utils.draw_heatmap_1d(np.log(mis_mean_resorted['mi_mean']), mis_mean_resorted['electrodes'])
    
    # %% Test
    # data = read_functional_connectivity('sub1ex1_alpha', 'pcc')
    data_ = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc', 'alpha')