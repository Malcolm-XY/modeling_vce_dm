# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:30:25 2025

@author: usouu
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial
from scipy.stats import boxcox
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

import feature_engineering
import vce_modeling
import drawer_channel_weight

def apply_transform(x, method):
    x = x + 1e-6  # avoid log(0) or boxcox(0)
    if method == 'boxcox':
        x, _ = boxcox(x)
    elif method == 'sqrt':
        x = np.sqrt(x)
    elif method == 'log':
        x = np.log(x)
    elif method == 'none':
        pass
    else:
        raise ValueError(f"Unsupported transform_method: {method}")
    return x

def preprocessing_r_target(r, normalize_method, transform_method):
    r = feature_engineering.normalize_matrix(r, normalize_method)
    r = apply_transform(r, transform_method)

    return r

def preprocessing_r_fitting(r, normalize_method, transform_method, r_target, mean_align_method):
    r = feature_engineering.normalize_matrix(r, normalize_method)
    r = apply_transform(r, transform_method)

    if mean_align_method == 'match_mean':
        delta = np.mean(r_target) - np.mean(r)
        r += delta

    return r

def prepare_target_and_inputs(
    feature='PCC',
    ranking_method='label_driven_mi_origin',
    distance_method='euclidean',
    normalize_method='minmax',
    transform_method='boxcox',
    mean_align_method='match_mean',
):
    weights = drawer_channel_weight.get_ranking_weight(ranking_method)
    r_target = preprocessing_r_target(weights.to_numpy(), normalize_method, transform_method)

    _, distance_matrix = feature_engineering.compute_distance_matrix('seed', method=distance_method, stereo_params={'prominence': 0.5, 'epsilon': 0.01}, visualize=True)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)

    global_joint_average = vce_modeling.load_global_averages(feature=feature)
    connectivity_matrix = feature_engineering.normalize_matrix(global_joint_average)

    preprocessing_fn = partial(
        preprocessing_r_fitting,
        normalize_method=normalize_method,
        transform_method=transform_method,
        r_target=r_target,
        mean_align_method=mean_align_method
    )

    return r_target, distance_matrix, connectivity_matrix, preprocessing_fn

def compute_r_fitting(method, params_dict, distance_matrix, connectivity_matrix, preprocessing_fn):
    factor_matrix = vce_modeling.compute_volume_conduction_factors(distance_matrix, method=method, params=params_dict)
    factor_matrix = feature_engineering.normalize_matrix(factor_matrix)
    differ_PCC_DM = feature_engineering.normalize_matrix(connectivity_matrix - factor_matrix)
    r_fitting = np.mean(differ_PCC_DM, axis=0)
    r_fitting = feature_engineering.normalize_matrix(r_fitting)
    return preprocessing_fn(r_fitting)

def optimize_and_store(name, loss_fn, x0, bounds, param_keys, distance_matrix, connectivity_matrix, preprocessing_fn):
    res = minimize(loss_fn, x0=x0, bounds=bounds)
    params = dict(zip(param_keys, res.x))
    results[name] = {'params': params, 'loss': res.fun}
    fittings[name] = compute_r_fitting(name, params, distance_matrix, connectivity_matrix, preprocessing_fn)

def loss_fn_template(method_name, param_dict_fn, r_target, distance_matrix, connectivity_matrix, preprocessing_fn):
    def loss(params):
        return np.mean((compute_r_fitting(method_name, param_dict_fn(params), distance_matrix, connectivity_matrix, preprocessing_fn) - r_target) ** 2)
    return loss

if __name__ == '__main__':
    results = {}
    fittings = {}

    r_target, distance_matrix, connectivity_matrix, preprocessing_fn = prepare_target_and_inputs(
        feature='PCC',
        ranking_method='label_driven_mi_origin',
        distance_method='stereo',
        transform_method='boxcox',
    )

    # %% Validation of Fitting Comparison; Before Fitting
    # Electrode labels
    from utils import utils_feature_loading
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    if hasattr(electrodes, 'tolist'):
        electrodes = electrodes.tolist()
    
    x = electrodes  # Electrode names 作为横轴
    
    # Load and normalize non-modeled r
    r_non_fitted = vce_modeling.load_global_averages(feature='PCC')
    r_non_moldeled = np.mean(r_non_fitted, axis=0).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-2, 0))
    r_non_moldeled = scaler.fit_transform(r_non_moldeled).flatten()
    
    # Compute MSE
    from sklearn.metrics import mean_squared_error
    mse_nonmodeled = mean_squared_error(r_target, r_non_moldeled)
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(x, r_target, label='r_target', linestyle='--', marker='o', color='black')
    plt.plot(x, r_non_moldeled, label='r_non_modeled (before)', marker='x', linestyle=':')
    plt.title(f"Before Modeling - MSE: {mse_nonmodeled:.4f}")
    plt.xlabel("Electrodes")
    plt.ylabel("Importance (normalized)")
    plt.xticks(rotation=60)
    plt.tick_params(axis='x', labelsize=8)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # %% Fitting
    optimize_and_store('exponential', loss_fn_template('exponential', lambda p: {'sigma': p[0]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0], [(0.1, 20.0)], ['sigma'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('gaussian', loss_fn_template('gaussian', lambda p: {'sigma': p[0]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0], [(0.1, 20.0)], ['sigma'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('inverse', loss_fn_template('inverse', lambda p: {'sigma': p[0], 'alpha': p[1]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0, 2.0], [(0.1, 20.0), (0.1, 5.0)], ['sigma', 'alpha'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('powerlaw', loss_fn_template('powerlaw', lambda p: {'alpha': p[0]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0], [(0.1, 10.0)], ['alpha'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('rational_quadratic', loss_fn_template('rational_quadratic', lambda p: {'sigma': p[0], 'alpha': p[1]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0, 1.0], [(0.1, 20.0), (0.1, 10.0)], ['sigma', 'alpha'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('generalized_gaussian', loss_fn_template('generalized_gaussian', lambda p: {'sigma': p[0], 'beta': p[1]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0, 1.0], [(0.1, 20.0), (0.1, 5.0)], ['sigma', 'beta'], distance_matrix, connectivity_matrix, preprocessing_fn)
    optimize_and_store('sigmoid', loss_fn_template('sigmoid', lambda p: {'mu': p[0], 'beta': p[1]}, r_target, distance_matrix, connectivity_matrix, preprocessing_fn), [2.0, 1.0], [(0.1, 10.0), (0.1, 5.0)], ['mu', 'beta'], distance_matrix, connectivity_matrix, preprocessing_fn)

    print("=== Fitting Results of All Models (Minimum MSE) ===")
    for method, result in results.items():
        print(f"[{method.upper()}] Best Parameters: {result['params']}, Minimum MSE: {result['loss']:.6f}")

    # %% Validation of Fitting Comparison
    from utils import utils_feature_loading
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    if hasattr(electrodes, 'tolist'):
        electrodes = electrodes.tolist()
    
    x = electrodes  # Electrode names 作为横轴
    
    for method, r_fitting in fittings.items():
        plt.figure(figsize=(10, 4))  # 每张图单独设置大小
        plt.plot(x, r_target, label='r_target', linestyle='--', marker='o')
        plt.plot(x, r_fitting, label=f'r_fitting ({method})', marker='x')
        plt.title(f"{method.upper()} - MSE: {results[method]['loss']:.4f}")
        plt.xlabel("Electrodes")
        plt.ylabel("Importance (normalized)")
        plt.xticks(rotation=60)
        plt.tick_params(axis='x', labelsize=8)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # %% Validation of Heatmap
    heatmap_data = np.vstack([r_target] + [fittings[method] for method in fittings.keys()])
    heatmap_labels = ['target'] + list(fittings.keys())

    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap='viridis', cbar=True, xticklabels=False, yticklabels=heatmap_labels, linewidths=0.5, linecolor='gray')
    plt.title("Heatmap of r_target and All r_fitting Vectors")
    plt.xlabel("Channel Index")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()
    
    # %% Validation of Brain Topography
    from utils import utils_feature_loading
    electrodes = utils_feature_loading.read_distribution('seed')['channel']    
    # target
    r_target_ = r_target.copy()
    _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(r_target_, electrodes)
    drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    # non-fitted
    r_non_fitted = vce_modeling.load_global_averages(feature='PCC')
    r_non_fitted = np.mean(r_non_fitted, axis=0)
    _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(r_non_fitted, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    # fitted
    r_fitted_g_gaussian = fittings['generalized_gaussian']
    _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(r_fitted_g_gaussian, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    # r_fitted_inverse = fittings['inverse']
    # _, strength_ranked, in_original_indices = weight_map_drawer.rank_and_visualize_fc_strength(r_fitted_inverse, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    # weight_map_drawer.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    # r_fitted_sigmoid = fittings['sigmoid']
    # _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(r_fitted_sigmoid, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    # drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])

    # %% Resort Fittings; For Saving
    weights_sigmoid = fittings['sigmoid'].copy()
    from utils import utils_feature_loading
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    rank_sigmoid = {'weights': weights_sigmoid, 'electrodes': electrodes}
    # 获取排序后的索引（按权重降序）
    sorted_indices = np.argsort(-weights_sigmoid)  # 负号表示降序排序
    
    # 构造排序后的结果
    ranked_sigmoid = {
        'weights': weights_sigmoid[sorted_indices],               # 排序后的权重
        'electrodes': electrodes[sorted_indices],    # 排序后的电极
        'original_indices': sorted_indices               # 排序后电极在原始 electrodes 中的索引
    }
    
    weights_g_gaussian = fittings['generalized_gaussian'].copy()
    from utils import utils_feature_loading
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    rank_g_gaussian = {'weights': weights_sigmoid, 'electrodes': electrodes}
    # 获取排序后的索引（按权重降序）
    sorted_indices = np.argsort(-weights_g_gaussian)  # 负号表示降序排序
    
    # 构造排序后的结果
    ranked_g_gaussian = {
        'weights': weights_sigmoid[sorted_indices],               # 排序后的权重
        'electrodes': electrodes[sorted_indices],    # 排序后的电极
        'original_indices': sorted_indices               # 排序后电极在原始 electrodes 中的索引
    }