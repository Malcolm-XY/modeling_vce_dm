# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:11:18 2025

@author: 18307
"""

import os
import pickle
import numpy as np
import pandas as pd

from utils import utils_feature_loading
def compute_mean_functional_connectivity(feature, subject_range=range(1,16), experiment_range=range(1,4), band='joint'):
    band = band.lower()

    global data
    data = []
    for sub in subject_range:
        for ex in experiment_range:
            if band == 'joint':
                features = utils_feature_loading.read_fcs_mat(dataset='seed', identifier=f'sub{sub}ex{ex}', feature=feature)
                alpha = features['alpha']
                beta = features['beta']
                gamma = features['gamma']
                
                data_temp = (alpha + beta + gamma)/3
                
            elif band in ['alpha', 'beta', 'gamma']:
                features = utils_feature_loading.read_fcs(dataset='seed', identifier=f'sub{sub}ex{ex}', feature=feature)
                
                data_temp = features[band]
            
            data.append(data_temp)
    
    data = np.array(data)
    data_mean = data.mean(axis=(0,1,2))
    
    return data_mean

# %% Example Usage
if __name__ == '__main__':
    #feature_mean = compute_mean_functional_connectivity('pcc', subject_range=range(1,16), experiment_range=range(1,2))
    
    # feature_engineering
    import feature_engineering
    feature_mean_ = feature_engineering.compute_averaged_fcnetwork_mat('plv', subjects=range(1, 11), 
                                                                     experiments=range(1, 4), draw=True, save=False)
    
    import vce_modeling
    feature_mean__ = vce_modeling.load_global_averaged_mat(feature='plv')