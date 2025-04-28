# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:30:25 2025

@author: usouu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.optimize import differential_evolution

import feature_engineering
import vce_modeling
import drawer_channel_weight

# %% Normalize and prune CW
def prune_cw(cw, normalize_method='minmax', transform_method='boxcox'):
    cw = feature_engineering.normalize_matrix(cw, transform_method)
    cw = feature_engineering.normalize_matrix(cw, normalize_method)
    return cw

# %% Compute CM, get CW as Agent of GCM and RCM
from utils import utils_feature_loading, utils_visualization
def preprocessing_cm_global_averaged(cm_global_averaged, coordinates):
    # Global averaged connectivity matrix; For subsquent fitting computation
    cm_global_averaged = np.abs(cm_global_averaged)
    cm_global_averaged = feature_engineering.normalize_matrix(cm_global_averaged)
    
    # Rebuild CM; By removing bads and Gaussian smoothing
    param = {
    'method': 'zscore', 'threshold': 2.5,
    'kernel': 'gaussian',  # idw or 'gaussian'
    'sigma': 5.0,  # only used for gaussian
    'manual_bad_idx': []}
    
    cm_global_averaged = feature_engineering.rebuild_features(cm_global_averaged, coordinates, param, True)
    
    # 2D Gaussian Smooth CM
    # connectivity_matrix = gaussian_filter(connectivity_matrix, sigma=0.5)
    
    # Spatial Gaussian Smooth CM
    cm_global_averaged = feature_engineering.spatial_gaussian_smoothing_on_fc_matrix(cm_global_averaged, coordinates, 5, True)
    
    return cm_global_averaged

def prepare_target_and_inputs(feature='pcc', ranking_method='label_driven_mi_origin', idxs_manual_remove=None):
    """
    Prepares smoothed channel weights, distance matrix, and global averaged connectivity matrix,
    with optional removal of specified bad channels.

    Parameters
    ----------
    feature : str
        Connectivity feature type (e.g., 'PCC').
    ranking_method : str
        Method for computing channel importance weights.
    idxs_manual_remove : list of int or None
        Indices of channels to manually remove from all matrices/vectors.

    Returns
    -------
    cw_target_smooth : np.ndarray of shape (n,)
    distance_matrix : np.ndarray of shape (n, n)
    cm_global_averaged : np.ndarray of shape (n, n)
    """
    # === 0. Electrodes; Remove specified channels
    electrodes = np.array(utils_feature_loading.read_distribution('seed')['channel'])
    electrodes = feature_engineering.remove_idx_manual(electrodes, idxs_manual_remove)
    
    # === 1. Target channel weight
    global channel_weights
    channel_weights = drawer_channel_weight.get_ranking_weight(ranking_method)
    cw_target = prune_cw(channel_weights.to_numpy())
    # ==== 1.1 Remove specified channels
    cw_target = feature_engineering.remove_idx_manual(cw_target, idxs_manual_remove)
    # === 1.2 Coordinates and smoothing
    coordinates = utils_feature_loading.read_distribution('seed')
    coordinates = coordinates.drop(idxs_manual_remove)
    cw_target_smooth = feature_engineering.spatial_gaussian_smoothing_on_vector(cw_target, coordinates, 2.0)

    # === 2. Distance matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d"})
    # === 2.1 Remove specified channels
    distance_matrix = feature_engineering.remove_idx_manual(distance_matrix, idxs_manual_remove)
    # === 2.2 Normalization
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)

    # === 3. Connectivity matrix
    connectivity_matrix_global_joint_averaged = vce_modeling.load_global_averages(feature=feature)
    # === 3.1 Remove specified channels
    cm_global_averaged = feature_engineering.remove_idx_manual(connectivity_matrix_global_joint_averaged, idxs_manual_remove)
    # === 3.2 Smoothing
    cm_global_averaged = preprocessing_cm_global_averaged(cm_global_averaged, coordinates)

    return electrodes, cw_target_smooth, distance_matrix, cm_global_averaged

# Here utilized VCE Model/FM=M(DM) Model
def compute_cw_fitting(method, params_dict, distance_matrix, connectivity_matrix):
    # FM=M(DM); VCE Model
    factor_matrix = vce_modeling.compute_volume_conduction_factors_advanced_model(distance_matrix, method, params_dict)
    factor_matrix = feature_engineering.normalize_matrix(factor_matrix)
    
    # RCM; Difference between CM and FM
    cm, fm = connectivity_matrix, factor_matrix
    scale_a = params_dict['scale_a']
    scale_b = params_dict['scale_b']
    e = 1e-6
    cm_recovered = cm + scale_a*fm + scale_b*cm/(gaussian_filter(fm, sigma=1)+e)
    cm_recovered = feature_engineering.normalize_matrix(cm_recovered)
    
    # RCW; Calculate from RCM
    global cw_fitting
    cw_fitting = np.mean(cm_recovered, axis=0)
    cw_fitting = prune_cw(cw_fitting)
    return cw_fitting

# %% Optimization
def optimize_and_store(name, loss_fn, bounds, param_keys, distance_matrix, connectivity_matrix):
    res = differential_evolution(loss_fn, bounds=bounds, strategy='best1bin', maxiter=1000)
    params = dict(zip(param_keys, res.x))
    
    result = {'params': params, 'loss': res.fun}
    cw_fitting = compute_cw_fitting(name, params, distance_matrix, connectivity_matrix)
    
    return result, cw_fitting

def loss_fn_template(method_name, param_dict_fn, cw_target, distance_matrix, connectivity_matrix):
    def loss_fn(params):
        loss = np.mean((compute_cw_fitting(method_name, param_dict_fn(params), distance_matrix, connectivity_matrix) - cw_target) ** 2)
        return loss
    return loss_fn

def loss_fn_template_by_rank(method_name, param_dict_fn, cw_target, distance_matrix, connectivity_matrix):
    def loss_fn(params):
        cw_fitting = compute_cw_fitting(method_name, param_dict_fn(params), distance_matrix, connectivity_matrix)
        cw_rank_fitting = np.argsort(cw_fitting)
        cw_rank_target = np.argsort(cw_target)
        loss = np.mean((cw_rank_fitting - cw_rank_target) ** 2)
        return loss
    return loss_fn

def fitting_basic_model():
    results, cws_fitting = {}, {}
    
    fitting_method_config = {
        'exponential': {
            'param_names': ['sigma', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'sigma': p[0], 'scale_a': p[1], 'scale_b': p[2]
            }
        },
        'gaussian': {
            'param_names': ['sigma', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'sigma': p[0], 'scale_a': p[1], 'scale_b': p[2]
            }
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'sigma': p[0], 'alpha': p[1], 'scale_a': p[2], 'scale_b': p[3]
            }
        },
        'powerlaw': {
            'param_names': ['alpha', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'alpha': p[0], 'scale_a': p[1], 'scale_b': p[2]
            }
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'sigma': p[0], 'alpha': p[1], 'scale_a': p[2], 'scale_b': p[3]
            }
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'sigma': p[0], 'beta': p[1], 'scale_a': p[2], 'scale_b': p[3]
            }
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'mu': p[0], 'beta': p[1], 'scale_a': p[2], 'scale_b': p[3]
            }
        }
    }

    for method, config in fitting_method_config.items():
        print(method)
        
        results[method], cws_fitting[method] = optimize_and_store(
            method,
            loss_fn_template(method, config['param_func'], cw_target, distance_matrix, cm_global_averaged),
            config['bounds'],
            config['param_names'],
            distance_matrix, cm_global_averaged
        )
    
    print("=== Fitting Results of All Models (Minimum MSE) ===")
    for method, result in results.items():
        print(f"[{method.upper()}] Best Parameters: {result['params']}, Minimum MSE: {result['loss']:.6f}")

    return results, cws_fitting

def fitting_advanced_model():
    results, cws_fitting = {}, {}
    
    fitting_method_config = {
        'exponential': {
            'param_names': ['sigma', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'sigma': p[0], 'deviation': p[1], 'offset': p[2],
                'scale_a': p[3], 'scale_b': p[4]
            }
        },
        'gaussian': {
            'param_names': ['sigma', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'sigma': p[0], 'deviation': p[1], 'offset': p[2],
                'scale_a': p[3], 'scale_b': p[4]
            }
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'sigma': p[0], 'alpha': p[1], 'deviation': p[2], 'offset': p[3],
                'scale_a': p[4], 'scale_b': p[5]
            }
        },
        'powerlaw': {
            'param_names': ['alpha', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'alpha': p[0], 'deviation': p[1], 'offset': p[2],
                'scale_a': p[3], 'scale_b': p[4]
            }
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'sigma': p[0], 'alpha': p[1], 'deviation': p[2], 'offset': p[3],
                'scale_a': p[4], 'scale_b': p[5]
            }
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'sigma': p[0], 'beta': p[1], 'deviation': p[2], 'offset': p[3],
                'scale_a': p[4], 'scale_b': p[5]
            }
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
            'param_func': lambda p: {
                'mu': p[0], 'beta': p[1], 'deviation': p[2], 'offset': p[3],
                'scale_a': p[4], 'scale_b': p[5]
            }
        }
    }
    
    for method, config in fitting_method_config.items():
        print(method)
        
        results[method], cws_fitting[method] = optimize_and_store(
            method,
            loss_fn_template(method, config['param_func'], cw_target, distance_matrix, cm_global_averaged),
            # loss_fn_template_by_rank(method, config['param_func'], cw_target, distance_matrix, cm_global_averaged),
            config['bounds'],
            config['param_names'],
            distance_matrix, cm_global_averaged
        )
    
    print("=== Fitting Results of All Models (Minimum MSE) ===")
    for method, result in results.items():
        print(f"[{method.upper()}] Best Parameters: {result['params']}, Minimum MSE: {result['loss']:.6f}")
    
    return results, cws_fitting
    
# %% Visualization
def draw_scatter_comparison(x, A, B, pltlabels={'title':'title', 
                                                'label_x':'label_x', 'label_y':'label_y', 
                                                'label_A':'label_A', 'label_B':'label_B'}):
    # Compute MSE
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(A, B)
    
    # Labels
    title = pltlabels.get('title')
    label_x = pltlabels.get('label_x')
    label_y = pltlabels.get('label_y')
    label_A = pltlabels.get('label_A')
    label_B = pltlabels.get('label_B')
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(x, A, label=label_A, linestyle='--', marker='o', color='black')
    plt.plot(x, B, label=label_B, marker='x', linestyle=':')
    plt.title(f"{title} - MSE: {mse:.4f}")
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.xticks(rotation=60)
    plt.tick_params(axis='x', labelsize=8)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def sort_ams(ams, labels, original_labels=None):
    dict_ams = pd.DataFrame({'ams': ams, 'labels': labels})
    
    sorted_ams = dict_ams.sort_values(by='ams', ascending=False).reset_index()
            
    idxs_in_original = []
    for label in sorted_ams['labels']:
        idx_in_original = list(original_labels).index(label)
        idxs_in_original.append(idx_in_original)
    
    sorted_ams['idxs_in_original'] = idxs_in_original
    
    return sorted_ams

# %% Usage
if __name__ == '__main__':
    # Fittin target and DM
    channel_manual_remove = [57, 61] # or # channel_manual_remove = [57, 61, 58, 59, 60]
    electrodes, cw_target, distance_matrix, cm_global_averaged = prepare_target_and_inputs('PCC', 
                                                    'label_driven_mi_origin', channel_manual_remove)

    # %% Fitting
    # results, cws_fitting = fitting_basic_model()
    results, cws_fitting = fitting_advanced_model()

    # %% Validation of Fitting Comparison
    cw_non_modeled = np.mean(cm_global_averaged, axis=0)
    cw_non_modeled = feature_engineering.normalize_matrix(cw_non_modeled)
    
    pltlabels = {'title':'Comparison of CWs; Before Modeling',
                 'label_x':'Electrodes', 'label_y':'Channel Weight', 
                 'label_A':'CW_LD_MI(Target)', 'label_B':'CW_CM_PCC(Non fitted)'}
    
    draw_scatter_comparison(electrodes, cw_target, cw_non_modeled, pltlabels)
    
    pltlabels = {'title':'Comparison of CWs; Before Modeling',
                 'label_x':'Electrodes', 'label_y':'Channel Weight', 
                 'label_A':'CW_LD_MI(Target)', 'label_B':'CW_CM_PCC(Non fitted)'}

    for method, cw_fitting in cws_fitting.items():
        _pltlabels = pltlabels.copy()
        _pltlabels['title'] = f'Comparison of CWs; {method}'
        _pltlabels['label_B'] = 'CW_Recovered_CM_PCC(Fitted)'
        draw_scatter_comparison(electrodes, cw_target, cw_fitting, _pltlabels)
    
    # %% Validation of Brain Topography
    # Coordinates
    coordinates = utils_feature_loading.read_distribution('seed')
    coordinates = coordinates.drop(index=channel_manual_remove)

    # target
    drawer_channel_weight.draw_2d_mapping(cw_target, coordinates, electrodes, 'Target: Channel Weights of LD_MI')
    
    # non-modeled
    drawer_channel_weight.draw_2d_mapping(cw_non_modeled, coordinates, electrodes, 'Non-Modeled: Channel Weights of CM_PCC')
    
    # fitted
    for method, cw_fitted in cws_fitting.items():
        drawer_channel_weight.draw_2d_mapping(cw_fitted, coordinates, electrodes, f'{method}_Modeled(Fitted)')
        
    # %% Validation of Heatmap
    cws_fitting['cw_target'] = cw_target
    utils_visualization.draw_joint_heatmap_1d(cws_fitting)
    
    # %% Sort ranks of channel weights based on fitted models
    # electrodes
    electrodes_original = np.array(utils_feature_loading.read_distribution('seed')['channel'])
    
    # target
    cw_target = feature_engineering.insert_idx_manual(cw_target, channel_manual_remove, value=0)
    sorted_cw_target = sort_ams(cw_target, electrodes_original, electrodes_original)
    
    # non-modeled
    cw_non_modeled = feature_engineering.insert_idx_manual(cw_non_modeled, channel_manual_remove, value=0)
    sorted_cw_non_modeled = sort_ams(cw_non_modeled, electrodes_original, electrodes_original)
    
    # fitted
    cws_fitted = {}
    cws_sorted = {}
    for method, cw_fitted in cws_fitting.items():
        cw_fitted_temp = feature_engineering.insert_idx_manual(cws_fitting[method], channel_manual_remove, value=0)
        cws_fitted[method] = cw_fitted_temp
        cw_sorted_temp = sort_ams(cw_fitted_temp, electrodes_original, electrodes_original)
        cws_sorted[method] = cw_sorted_temp
    