# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:21:23 2025

@author: usouu
"""
import os
import numpy as np

import torch
import feature_engineering
import cnn_validation
from models import models
from utils import utils_feature_loading

import cw_manager

def cnn_evaluation_circle_original_cm(feature_cm, subject_range=range(1, 6), experiment_range=range(1, 4), save=False):
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed')
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_original = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']
            
            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']
            
            x = np.stack((alpha, beta, gamma), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_CM = cnn_validation.cnn_cross_validation(cnn_model, x, y)
            
            # Add identifier to the result
            result_CM['Identifier'] = f'sub{sub}ex{ex}'
            
            all_results_original.append(result_CM)
    
    # save
    output_dir = os.path.join(os.getcwd(), 'results_cnn_evaluation')
    filename_CM = "cnn_validation_CM_PCC.xlsx"
    if save: save_results_to_xlsx_append(all_results_original, output_dir, filename_CM)
    
    return all_results_original