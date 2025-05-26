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
from cnn_val_circle import read_params
from cnn_val_circle import save_results_to_xlsx_append

def cnn_subnetworks_evaluation_circle_control_1(argument='data_driven_pcc_10_15', selection_rate=1, feature_cm='pcc', 
                                              subject_range=range(11,16), experiment_range=range(1,4), 
                                              save=False):
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed')
    y = torch.tensor(np.array(labels)).view(-1)
        
    channel_weights = cw_manager.read_channel_weight_DD(argument, sort=True)
    channel_selected = channel_weights.index[:int(len(channel_weights.index)*selection_rate)]
    
    # data and evaluation circle
    all_results_original = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            alpha = features['alpha'][:,channel_selected,:][:,:,channel_selected]
            beta = features['beta'][:,channel_selected,:][:,:,channel_selected]
            gamma = features['gamma'][:,channel_selected,:][:,:,channel_selected]
            
            # CM/H5
            # features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            # alpha = features['alpha'][:,channel_selected,:][:,:,channel_selected]
            # beta = features['beta'][:,channel_selected,:][:,:,channel_selected]
            # gamma = features['gamma'][:,channel_selected,:][:,:,channel_selected]

            x = np.stack((alpha, beta, gamma), axis=1)

            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_CM = cnn_validation.cnn_cross_validation(cnn_model, x, y)
            
            # Add identifier to the result
            result_CM['Identifier'] = f'sub{sub}ex{ex}'
            
            all_results_original.append(result_CM)

    # save
    output_dir = os.path.join(os.getcwd(), 'results_cnn_subnetworks_evaluation')
    filename_CM = "cnn_validation_CM_PCC.xlsx"
    if save: save_results_to_xlsx_append(all_results_original, output_dir, filename_CM)
    
    return all_results_original

def cnn_subnetworks_evaluation_circle_control_2(argument='label_driven_mi_10_15', selection_rate=1, feature_cm='pcc', 
                                              subject_range=range(11,16), experiment_range=range(1,4), 
                                              save=False):
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed')
    y = torch.tensor(np.array(labels)).view(-1)
        
    channel_weights = cw_manager.read_channel_weight_DD(argument, sort=True)
    channel_selected = channel_weights.index[:int(len(channel_weights.index)*selection_rate)]
    
    # data and evaluation circle
    all_results_original = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            alpha = features['alpha'][:,channel_selected,:][:,:,channel_selected]
            beta = features['beta'][:,channel_selected,:][:,:,channel_selected]
            gamma = features['gamma'][:,channel_selected,:][:,:,channel_selected]
            
            # CM/H5
            # features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            # alpha = features['alpha'][:,channel_selected,:][:,:,channel_selected]
            # beta = features['beta'][:,channel_selected,:][:,:,channel_selected]
            # gamma = features['gamma'][:,channel_selected,:][:,:,channel_selected]

            x = np.stack((alpha, beta, gamma), axis=1)

            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_CM = cnn_validation.cnn_cross_validation(cnn_model, x, y)
            
            # Add identifier to the result
            result_CM['Identifier'] = f'sub{sub}ex{ex}'
            
            all_results_original.append(result_CM)

    # save
    output_dir = os.path.join(os.getcwd(), 'results_cnn_subnetworks_evaluation')
    filename_CM = "cnn_validation_CM_PCC.xlsx"
    if save: save_results_to_xlsx_append(all_results_original, output_dir, filename_CM)
    
    return all_results_original

from connectivity_matrix_rebuilding import cm_rebuilding as cm_rebuild
def cnn_subnetworks_evaluation_circle_rebuilt_cm(model, model_fm, model_rcm, 
                                                 argument='fitting_results(10_15_joint_band_from_mat)', 
                                                 selection_rate=1, feature_cm='pcc', 
                                                 subject_range=range(11,16), experiment_range=range(1,4), 
                                                 save=False):
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed')
    y = torch.tensor(np.array(labels)).view(-1)
    
    # experiment; channel weights computed from the rebuilded connectivity matrix that constructed by vce modeling
    channel_weights = cw_manager.read_channel_weight_fitting(model_fm, model_rcm, model, 
                                    source=argument, sort=True)
    channel_selected = channel_weights.index[:int(len(channel_weights.index)*selection_rate)]
    
    # distance matrix
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d"}, visualize=True)
    dm = feature_engineering.normalize_matrix(dm)
    
    # parameters for construction of FM and RCM
    param = read_params(model, model_fm, model_rcm, folder=argument)
    
    # data and evaluation circle
    all_results_rebuilded = []
    average_accuracy_rebuilded, average_accuracy_rebuilded_counter = 0.0, 0
    
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # CM/H5
            # features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # RCM
            alpha_rebuilded = cm_rebuild(alpha, dm, param, model, model_fm, model_rcm, True, False)
            beta_rebuilded = cm_rebuild(beta, dm, param, model, model_fm, model_rcm, True, False)
            gamma_rebuilded = cm_rebuild(gamma, dm, param, model, model_fm, model_rcm, True, False)
            
            # subnetworks
            alpha_rebuilded = alpha_rebuilded[:,channel_selected,:][:,:,channel_selected]
            beta_rebuilded = beta_rebuilded[:,channel_selected,:][:,:,channel_selected]
            gamma_rebuilded = gamma_rebuilded[:,channel_selected,:][:,:,channel_selected]
            
            x_rebuilded = np.stack((alpha_rebuilded, beta_rebuilded, gamma_rebuilded), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuilded, y)
            
            # Add identifier to the result
            result_RCM['Identifier'] = f'sub{sub}ex{ex}'
            all_results_rebuilded.append(result_RCM)
            
            average_accuracy_rebuilded += result_RCM['accuracy']
            average_accuracy_rebuilded_counter += 1

    average_accuracy_rebuilded = {'accuracy': average_accuracy_rebuilded/average_accuracy_rebuilded_counter}
    all_results_rebuilded.append(average_accuracy_rebuilded)
    
    # print(f'Final Results: {results_entry}')
    print('K-Fold Validation compelete\n')
    
    # save
    output_dir = os.path.join(os.getcwd(), 'results_cnn_subnetworks_evaluation')
    
    identifier = f'{model_fm.lower()}_fm_{model_rcm.lower()}_rcm'
    filename_RCM = f"cnn_validation_RCM({identifier})_{model}_{feature_cm}.xlsx"
    
    if save: save_results_to_xlsx_append(all_results_rebuilded, output_dir, filename_RCM)
    
    return all_results_rebuilded

import pandas as pd
def cnn_subnetworks_eval_circle_rcm_intergrated(model_fm, model_rcm, selection_rate, feature_cm, save=False):
    model = list(['exponential', 'gaussian', 'inverse', 'powerlaw', 'rational_quadratic', 'generalized_gaussian', 'sigmoid'])
    
    avgs_results_fitting = []
    for trail in range(0, 7):
        results_fitting, avg_results_fitting = cnn_subnetworks_evaluation_circle_rebuilt_cm(model[trail], model_fm, model_rcm, 
                                                                          selection_rate=selection_rate, feature_cm=feature_cm,
                                                                          save=save) # save=True)
        avg_results_fitting = np.array([model[trail], avg_results_fitting['accuracy']])
        avgs_results_fitting.append(avg_results_fitting)
        
    avgs_results_fitting = np.vstack(avgs_results_fitting)
    avgs_results_fitting_df = pd.DataFrame(avgs_results_fitting)
    
    return avgs_results_fitting, avgs_results_fitting_df

if __name__ == '__main__':
    cnn_subnetworks_evaluation_circle_control_1(selection_rate=0.25, feature_cm='pcc', save=True)
    cnn_subnetworks_evaluation_circle_control_1(selection_rate=0.25, feature_cm='pcc', save=True)
    cnn_subnetworks_evaluation_circle_rebuilt_cm(model='exponential', model_fm='advanced', model_rcm='linear', 
                                                     argument='fitting_results(10_15_joint_band_from_mat)', 
                                                     selection_rate=0.25, feature_cm='pcc', 
                                                     subject_range=range(11,16), experiment_range=range(1,4), 
                                                     save=True)
