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
def compute_cw_fitting(method, params_dict, distance_matrix, connectivity_matrix, RCM='differ'):
    """
    Compute cw_fitting based on selected RCM method: differ, linear, or linear_ratio.
    """
    RCM = RCM.lower()

    # Step 1: Calculate FM
    factor_matrix = vce_modeling.compute_volume_conduction_factors_advanced_model(distance_matrix, method, params_dict)
    factor_matrix = feature_engineering.normalize_matrix(factor_matrix)

    # Step 2: Calculate RCM
    cm, fm = connectivity_matrix, factor_matrix
    e = 1e-6  # Small value to prevent division by zero

    if RCM == 'differ':
        cm_recovered = cm - fm
    elif RCM == 'linear':
        scale_a = params_dict.get('scale_a', 1.0)
        cm_recovered = cm + scale_a * fm
    elif RCM == 'linear_ratio':
        scale_a = params_dict.get('scale_a', 1.0)
        scale_b = params_dict.get('scale_b', 1.0)
        cm_recovered = cm + scale_a * fm + scale_b * cm / (gaussian_filter(fm, sigma=1) + e)
    else:
        raise ValueError(f"Unsupported RCM mode: {RCM}")

    # Step 3: Normalize RCM
    cm_recovered = feature_engineering.normalize_matrix(cm_recovered)

    # Step 4: Compute CW
    global cw_fitting
    cw_fitting = np.mean(cm_recovered, axis=0)
    cw_fitting = prune_cw(cw_fitting)

    return cw_fitting

# %% Optimization
def optimize_and_store(method, loss_fn, bounds, param_keys, distance_matrix, connectivity_matrix, RCM='differ'):
    res = differential_evolution(loss_fn, bounds=bounds, strategy='best1bin', maxiter=1000)
    params = dict(zip(param_keys, res.x))
    
    result = {'params': params, 'loss': res.fun}
    cw_fitting = compute_cw_fitting(method, params, distance_matrix, connectivity_matrix, RCM)
    
    return result, cw_fitting

def loss_fn_template(method_name, param_dict_fn, cw_target, distance_matrix, connectivity_matrix, RCM):
    def loss_fn(params):
        loss = np.mean((compute_cw_fitting(method_name, param_dict_fn(params), distance_matrix, connectivity_matrix, RCM) - cw_target) ** 2)
        return loss
    return loss_fn

class FittingConfig:
    """
    Configuration for fitting models.
    Provides param_names, bounds, and automatic param_func.
    """
    
    @staticmethod
    def get_config(model_type: str, recovery_type: str):
        """
        Get the config dictionary based on model type and recovery type.
    
        Args:
            model_type (str): 'basic' or 'advanced'
            recovery_type (str): 'differ', 'linear', or 'linear_ratio'
    
        Returns:
            dict: Corresponding config dictionary
    
        Raises:
            ValueError: If input type is invalid
        """
        model_type = model_type.lower()
        recovery_type = recovery_type.lower()
    
        if model_type == 'basic' and recovery_type == 'differ':
            return FittingConfig.config_basic_model_differ_recovery
        elif model_type == 'advanced' and recovery_type == 'differ':
            return FittingConfig.config_advanced_model_differ_recovery
        elif model_type == 'basic' and recovery_type == 'linear':
            return FittingConfig.config_basic_model_linear_recovery
        elif model_type == 'advanced' and recovery_type == 'linear':
            return FittingConfig.config_advanced_model_linear_recovery
        elif model_type == 'basic' and recovery_type == 'linear_ratio':
            return FittingConfig.config_basic_model_linear_ratio_recovery
        elif model_type == 'advanced' and recovery_type == 'linear_ratio':
            return FittingConfig.config_advanced_model_linear_ratio_recovery
        else:
            raise ValueError(f"Invalid model_type '{model_type}' or recovery_type '{recovery_type}'")
    
    @staticmethod
    def make_param_func(param_names):
        """Auto-generate param_func based on param_names."""
        return lambda p: {name: p[i] for i, name in enumerate(param_names)}

    config_basic_model_differ_recovery = {
        'exponential': {
            'param_names': ['sigma'],
            'bounds': [(0.1, 20.0)],
        },
        'gaussian': {
            'param_names': ['sigma'],
            'bounds': [(0.1, 20.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha'],
            'bounds': [(0.1, 20.0), (0.1, 5.0)],
        },
        'powerlaw': {
            'param_names': ['alpha'],
            'bounds': [(0.1, 10.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha'],
            'bounds': [(0.1, 20.0), (0.1, 10.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta'],
            'bounds': [(0.1, 20.0), (0.1, 5.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta'],
            'bounds': [(0.1, 10.0), (0.1, 5.0)],
        },
    }

    config_advanced_model_differ_recovery = {
        'exponential': {
            'param_names': ['sigma', 'deviation', 'offset'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'gaussian': {
            'param_names': ['sigma', 'deviation', 'offset'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0)],
        },
        'powerlaw': {
            'param_names': ['alpha', 'deviation', 'offset'],
            'bounds': [(0.1, 10.0), (1e-6, 1.0), (-1.0, 1.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'deviation', 'offset'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'deviation', 'offset'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
    }

    config_basic_model_linear_recovery = {
        'exponential': {
            'param_names': ['sigma', 'scale_a'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0)],
        },
        'gaussian': {
            'param_names': ['sigma', 'scale_a'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0)],
        },
        'powerlaw': {
            'param_names': ['alpha', 'scale_a'],
            'bounds': [(0.1, 10.0), (-1.0, 1.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'scale_a'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0)],
        },
    }

    config_advanced_model_linear_recovery = {
        'exponential': {
            'param_names': ['sigma', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'gaussian': {
            'param_names': ['sigma', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'powerlaw': {
            'param_names': ['alpha', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 10.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'deviation', 'offset', 'scale_a'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        },
    }

    config_basic_model_linear_ratio_recovery = {
        'exponential': {
            'param_names': ['sigma', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'gaussian': {
            'param_names': ['sigma', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'powerlaw': {
            'param_names': ['alpha', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
        },
    }

    config_advanced_model_linear_ratio_recovery = {
        'exponential': {
            'param_names': ['sigma', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'gaussian': {
            'param_names': ['sigma', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'inverse': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'powerlaw': {
            'param_names': ['alpha', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'rational_quadratic': {
            'param_names': ['sigma', 'alpha', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'generalized_gaussian': {
            'param_names': ['sigma', 'beta', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 20.0), (0.1, 5.0), (1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
        'sigmoid': {
            'param_names': ['mu', 'beta', 'deviation', 'offset', 'scale_a', 'scale_b'],
            'bounds': [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 2.0)],
        },
    }

def fitting_model(model_type='basic', recovery_type='differ', cw_target=None, distance_matrix=None, connectivity_matrix=None):
    """
    Perform model fitting across multiple methods.

    Args:
        model_type (str): 'basic' or 'advanced'
        recovery_type (str): 'differ', 'linear', 'linear_ratio'
        cw_target (np.ndarray): Target feature vector
        distance_matrix (np.ndarray): Distance matrix
        connectivity_matrix (np.ndarray): Connectivity matrix

    Returns:
        results (dict): Optimized parameters and losses
        cws_fitting (dict): Fitted CW vectors
    """

    results, cws_fitting = {}, {}

    # Load fitting configuration
    fitting_config = FittingConfig.get_config(model_type, recovery_type)

    for method, config in fitting_config.items():
        print(f"Fitting Method: {method}")

        param_names = config['param_names']
        bounds = config['bounds']
        param_func = FittingConfig.make_param_func(param_names)

        # Build loss function
        loss_fn = loss_fn_template(method, param_func, cw_target, distance_matrix, connectivity_matrix, RCM=recovery_type)

        # Optimize
        try:
            results[method], cws_fitting[method] = optimize_and_store(
                method,
                loss_fn,
                bounds,
                param_names,
                distance_matrix,
                connectivity_matrix,
                RCM=recovery_type
            )
        except Exception as e:
            print(f"[{method.upper()}] Optimization failed: {e}")
            results[method], cws_fitting[method] = None, None

    print("\n=== Fitting Results of All Models (Minimum MSE) ===")
    for method, result in results.items():
        if result is not None:
            print(f"[{method.upper()}] Best Parameters: {result['params']}, Minimum MSE: {result['loss']:.6f}")
        else:
            print(f"[{method.upper()}] Optimization Failed.")

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

def draw_joint_scatter(cws_fitting, cw_target, electrodes, ncols=3):
    """
    Draw scatter comparisons between target and each fitted CW.
    """
    from sklearn.metrics import mean_squared_error

    n_models = len(cws_fitting)
    nrows = (n_models + ncols - 1) // ncols

    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for i, (method, cw) in enumerate(cws_fitting.items(), 1):
        if cw is None:
            continue
        plt.subplot(nrows, ncols, i)
        mse = mean_squared_error(cw_target, cw)
        plt.plot(electrodes, cw_target, label='Target', linestyle='--', marker='o', color='black')
        plt.plot(electrodes, cw, label=f'{method}', marker='x', linestyle=':')
        plt.title(f'{method.upper()} - MSE: {mse:.4f}')
        plt.xlabel('Electrodes')
        plt.ylabel('Channel Weight')
        plt.xticks(rotation=60)
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def sort_ams(ams, labels, original_labels=None):
    dict_ams_original = pd.DataFrame({'labels': labels, 'ams': ams})
    
    dict_ams_sorted = dict_ams_original.sort_values(by='ams', ascending=False).reset_index()
            
    # idxs_in_original = []
    # for label in dict_ams_sorted['labels']:
    #     idx_in_original = list(original_labels).index(label)
    #     idxs_in_original.append(idx_in_original)
    
    dict_ams_summary = dict_ams_original.copy()
    # dict_ams_summary['idex_in_original'] = idxs_in_original
    
    dict_ams_summary = pd.concat([dict_ams_summary, dict_ams_sorted], axis=1)
    
    return dict_ams_summary

# %% Save
import os
def save_fitting_results(results, save_dir='results', file_name='fitting_results.xlsx'):
    """
    Save fitting results (parameters and losses) into an Excel or TXT file.
    """
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, file_name)
    
    # Organize results into DataFrame
    data = []
    for method, result in results.items():
        if result is None:
            continue
        row = {'method': method.upper()}
        row.update(result['params'])
        row['loss'] = result['loss']
        data.append(row)
    
    df = pd.DataFrame(data)

    # Save
    if file_name.endswith('.xlsx'):
        df.to_excel(results_path, index=False)
    elif file_name.endswith('.txt'):
        df.to_csv(results_path, sep='\t', index=False)
    else:
        raise ValueError("Unsupported file extension. Use .xlsx or .txt")

    print(f"Fitting results saved to {results_path}")

def save_channel_weights(cws_fitting, save_dir='results', file_name='channel_weights.xlsx'):
    """
    将包含多个 DataFrame 的字典保存为一个 Excel 文件，不同的 sheet 存储不同的 DataFrame。

    Args:
        cws_fitting (dict): 键是 sheet 名，值是 DataFrame 或可以转换成 DataFrame 的数据结构。
        save_dir (str): 保存目录，默认为 'results'。
        file_name (str): 保存的文件名，默认为 'channel_weights.xlsx'。
    """

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 组合完整路径
    save_path = os.path.join(save_dir, file_name)

    # 写入Excel
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        for sheet_name, data in cws_fitting.items():
            # 安全处理sheet名：截断长度，替换非法字符
            valid_sheet_name = sheet_name[:31].replace('/', '_').replace('\\', '_').replace('*', '_').replace('?', '_').replace(':', '_').replace('[', '_').replace(']', '_')

            # 如果data不是DataFrame，尝试转换
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

            # 写入sheet
            data.to_excel(writer, sheet_name=valid_sheet_name, index=False)

    print(f"Channel weights successfully saved to {save_path}")

# %% Usage
# %% Usage
if __name__ == '__main__':
    # Fittin target and DM
    channel_manual_remove = [57, 61] # or # channel_manual_remove = [57, 61, 58, 59, 60]
    electrodes, cw_target, distance_matrix, cm_global_averaged = prepare_target_and_inputs('PCC', 
                                                    'label_driven_mi_origin', channel_manual_remove)

    # %% Fitting
    results, cws_fitting = fitting_model('basic', 'differ', cw_target, distance_matrix, cm_global_averaged)
    
    # %% Sort ranks of channel weights based on fitted models
    # electrodes
    electrodes_original = np.array(utils_feature_loading.read_distribution('seed')['channel'])
    
    # target
    cw_target_rebuild = feature_engineering.insert_idx_manual(cw_target, channel_manual_remove, value=0)
    sorted_cw_target = sort_ams(cw_target_rebuild, electrodes_original, electrodes_original)
    
    # non-modeled
    cw_non_modeled = np.mean(cm_global_averaged, axis=0)
    cw_non_modeled = feature_engineering.normalize_matrix(cw_non_modeled)
    
    cw_non_modeled_rebuild = feature_engineering.insert_idx_manual(cw_non_modeled, channel_manual_remove, value=0)
    sorted_cw_non_modeled = sort_ams(cw_non_modeled_rebuild, electrodes_original, electrodes_original)
    
    # fitted
    cws_fitted = {}
    cws_sorted = {}
    for method, cw_fitted in cws_fitting.items():
        cw_fitted_temp = feature_engineering.insert_idx_manual(cws_fitting[method], channel_manual_remove, value=0)
        cws_fitted[method] = cw_fitted_temp
        cw_sorted_temp = sort_ams(cw_fitted_temp, electrodes_original, electrodes_original)
        cws_sorted[method] = cw_sorted_temp    
    
    # %% Save
    path_currebt = os.getcwd()
    results_path = os.path.join(os.getcwd(), 'fitting_results')
    save_fitting_results(results, results_path)
    save_channel_weights(cws_sorted, results_path)
    
    # %% Validation of Fitting Comparison
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
    
    # joint scatter #################
    draw_joint_scatter(cws_fitting, cw_target, electrodes)
    
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
    
    # joint topography #################
    
    # %% Validation of Heatmap
    cws_fitting['cw_target'] = cw_target
    utils_visualization.draw_joint_heatmap_1d(cws_fitting)
    