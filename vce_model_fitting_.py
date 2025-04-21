# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:30:25 2025

@author: usouu
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import boxcox, yeojohnson
from scipy.ndimage import gaussian_filter
from scipy.optimize import differential_evolution
from sklearn.preprocessing import MinMaxScaler

import feature_engineering
import vce_modeling
import drawer_channel_weight

# %% Normalize and prune CW
def normalize_array(x, method):
    x = x + 1e-6  # avoid log(0) or boxcox(0)
    if method == 'boxcox':
        x = boxcox(x, lmbda=0.75)
    elif method == 'yeojohnson':
        x, _ = yeojohnson(x)  # Yeo-Johnson support 0 and minus
    elif method == 'sqrt':
        x = np.sqrt(x)
    elif method == 'log':
        x = np.log(x)
    elif method == 'none':
        pass
    else:
        raise ValueError(f"Unsupported transform_method: {method}")
    return x

def prune_cw(cw, normalize_method='minmax', transform_method='boxcox'):
    cw = feature_engineering.normalize_matrix(cw, normalize_method)
    cw = normalize_array(cw, transform_method)
    cw = cw - np.mean(cw)
    return cw

# %% Compute CM, get CW as Agent of GCM and RCM
from scipy.spatial.distance import cdist
def spatial_gaussian_smoothing(A, coordinates, sigma):
    coords = np.vstack([coordinates['x'], coordinates['y'], coordinates['z']]).T
    dists = cdist(coords, coords)
    weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))
    weights /= weights.sum(axis=1, keepdims=True)
    A_smooth = weights @ A
    return A_smooth

def spatial_smoothing_on_fc_matrix(A, coordinates, sigma):
    """
    Applies spatial Gaussian smoothing to a symmetric functional connectivity (FC) matrix.

    Parameters
    ----------
    A : np.ndarray of shape (N, N)
        Symmetric functional connectivity matrix.
    coordinates : dict with keys 'x', 'y', 'z'
        Each value is a list or array of length N, giving 3D coordinates for each channel.
    sigma : float
        Standard deviation of the spatial Gaussian kernel.

    Returns
    -------
    A_smooth : np.ndarray of shape (N, N)
        Symmetrically smoothed functional connectivity matrix.
    """

    # Step 1: Stack coordinate vectors to (N, 3)
    coords = np.vstack([coordinates['x'], coordinates['y'], coordinates['z']]).T  # shape (N, 3)

    # Step 2: Compute Euclidean distance matrix between channels
    dists = cdist(coords, coords)  # shape (N, N)

    # Step 3: Compute spatial Gaussian weights
    weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))  # shape (N, N)
    weights /= weights.sum(axis=1, keepdims=True)       # normalize per row

    # Step 4: Apply spatial smoothing to both rows and columns
    A_smooth = weights @ A @ weights.T

    # Step 5 (optional): Enforce symmetry
    A_smooth = 0.5 * (A_smooth + A_smooth.T)

    return A_smooth

def remove_bads_and_rebuild_cm_IDW(CM, coordinates, param):
    n_channels = CM.shape[0]
    coords = np.vstack([coordinates['x'], coordinates['y'], coordinates['z']]).T

    # === 1. 自动检测 bad 通道
    mean_conn = np.mean(np.abs(CM), axis=1)

    if param['method'] == 'zscore':
        z = (mean_conn - np.mean(mean_conn)) / np.std(mean_conn)
        bad_idx = np.where(np.abs(z) > param['threshold'])[0]
    elif param['method'] == 'iqr':
        q1, q3 = np.percentile(mean_conn, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - param['threshold'] * iqr, q3 + param['threshold'] * iqr
        bad_idx = np.where((mean_conn < lower) | (mean_conn > upper))[0]
    else:
        raise ValueError("param['method'] must be 'zscore' or 'iqr'")

    # === 2. 手动指定 bad 通道
    manual_bad_idx = param.get('manual_bad_idx', [])
    manual_bad_idx = np.array(manual_bad_idx, dtype=int)

    # === 3. 合并 & 去重
    bad_idx = np.unique(np.concatenate([bad_idx, manual_bad_idx]))
    print(f"Detected bad channels: {bad_idx.tolist()}")

    if len(bad_idx) == 0:
        return CM.copy()

    CM_rebuild = CM.copy()

    if param['kernel'] == 'gaussian':
        sigma = param.get('sigma', 1.0)
        CM_smooth = spatial_smoothing_on_fc_matrix(CM, coordinates, sigma)
        for i in bad_idx:
            CM_rebuild[i, :] = CM_smooth[i, :]
            CM_rebuild[:, i] = CM_smooth[:, i]
    elif param['kernel'] == 'idw':
        for i in bad_idx:
            dists = np.linalg.norm(coords[i] - coords, axis=1)
            dists[i] = np.inf
            weights = 1 / dists
            weights[i] = 0
            weights /= weights.sum()
            CM_rebuild[i, :] = weights @ CM
            CM_rebuild[:, i] = CM_rebuild[i, :]
    else:
        raise ValueError("Unsupported kernel type")

    return CM_rebuild

def preprocessing_global_averaged_cm(feature='PCC'):
    # Global averaged connectivity matrix; For subsquent fitting computation
    global_joint_average = vce_modeling.load_global_averages(feature=feature)
    global_joint_average = np.abs(global_joint_average)
    connectivity_matrix = feature_engineering.normalize_matrix(global_joint_average)
    
    # Apply Spatial Gaussian Smoothing to CM
    # connectivity_matrix = gaussian_filter(connectivity_matrix, sigma=0.5)
    
    # Apply Spatial Gaussian Smoothing for CM to CM
    from utils import utils_feature_loading, utils_visualization
    utils_visualization.draw_projection(connectivity_matrix)
    
    coordinates = utils_feature_loading.read_distribution('seed')
    param = {
    'method': 'zscore',
    'threshold': 2.5,
    'kernel': 'idw',  # idw or 'gaussian'
    'sigma': 0.5,  # only used for gaussian
    'manual_bad_idx': []
    }
    connectivity_matrix = remove_bads_and_rebuild_cm_IDW(connectivity_matrix, coordinates, param)
    
    utils_visualization.draw_projection(connectivity_matrix)
    
    return connectivity_matrix

def boxcox_transform_matrix(matrix, epsilon=1e-6):
    """
    Applies Box-Cox transformation to each column of a 2D matrix.

    Parameters:
        matrix (np.ndarray): 2D array of shape (n_samples, n_features). All values must be >= 0.
        epsilon (float): Small constant added to avoid log(0), default=1e-6.

    Returns:
        transformed_matrix (np.ndarray): Box-Cox transformed matrix of same shape.
        lambdas (list): List of lambda values used for each column.
    """
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    
    matrix = matrix + epsilon  # Avoid zero values
    transformed_matrix = np.zeros_like(matrix)
    lambdas = []

    for i in range(matrix.shape[1]):
        transformed_col, lam = boxcox(matrix[:, i])
        transformed_matrix[:, i] = transformed_col
        lambdas.append(lam)

    return transformed_matrix, lambdas

def prepare_target_and_inputs(feature='PCC', ranking_method='label_driven_mi_origin', manual_bad_idx=None):
    """
    Prepares smoothed channel weights, distance matrix, and global averaged connectivity matrix,
    with optional removal of specified bad channels.

    Parameters
    ----------
    feature : str
        Connectivity feature type (e.g., 'PCC').
    ranking_method : str
        Method for computing channel importance weights.
    manual_bad_idx : list of int or None
        Indices of channels to manually remove from all matrices/vectors.

    Returns
    -------
    cw_target_smooth : np.ndarray of shape (n,)
    distance_matrix : np.ndarray of shape (n, n)
    connectivity_matrix : np.ndarray of shape (n, n)
    """
    import numpy as np
    from utils import utils_feature_loading

    # === 1. Target channel weight
    weights = drawer_channel_weight.get_ranking_weight(ranking_method)
    cw_target = prune_cw(weights.to_numpy())

    # === 2. Coordinates for smoothing
    coordinates = utils_feature_loading.read_distribution('seed')
    cw_target_smooth = spatial_gaussian_smoothing(cw_target, coordinates, sigma=20.0)

    # === 3. Distance matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix(
        dataset="SEED", projection_params={"type": "3d"})
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)

    # === 4. Connectivity matrix
    connectivity_matrix = preprocessing_global_averaged_cm(feature=feature)

    # === 5. Remove specified bad channels
    if manual_bad_idx is not None and len(manual_bad_idx) > 0:
        manual_bad_idx = sorted(set(manual_bad_idx))
        print(f"Manually removing bad channels: {manual_bad_idx}")

        mask = np.ones(cw_target_smooth.shape[0], dtype=bool)
        mask[manual_bad_idx] = False

        cw_target_smooth = cw_target_smooth[mask]
        distance_matrix = distance_matrix[mask][:, mask]
        connectivity_matrix = connectivity_matrix[mask][:, mask]

    return cw_target_smooth, distance_matrix, connectivity_matrix

def prepare_target_and_inputs_(feature='PCC', ranking_method='label_driven_mi_origin'):
    # Target; Label-Driven CW
    weights = drawer_channel_weight.get_ranking_weight('label_driven_mi_origin')
    cw_target = prune_cw(weights.to_numpy())
    
    # Apply Spatial Gaussian smoothing to target
    from utils import utils_feature_loading
    coordinates = utils_feature_loading.read_distribution('seed')
    cw_target_smooth = spatial_gaussian_smoothing(cw_target, coordinates, sigma=20.0)
    
    # Distance matrix; For subsquent fitting computation
    # _, distance_matrix = feature_engineering.compute_distance_matrix('seed',
    #     projection_params={'type': 'azimuthal', 'y_compression_factor': 1,'y_compression_direction': 'negative'})
    
    _, distance_matrix = feature_engineering.compute_distance_matrix(dataset="SEED", projection_params={"type": "3d"})
    
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)

    # Global averaged connectivity matrix; For subsquent fitting computation
    connectivity_matrix = preprocessing_global_averaged_cm(feature=feature)

    return cw_target_smooth, distance_matrix, connectivity_matrix

def compute_cw_fitting(method, params_dict, distance_matrix, connectivity_matrix):
    # FM; VCE Model
    factor_matrix = vce_modeling.compute_volume_conduction_factors(distance_matrix, method=method, params=params_dict)
    factor_matrix = feature_engineering.normalize_matrix(factor_matrix)
    
    # RCM; Difference between CM and FM
    cm, fm = connectivity_matrix, factor_matrix
    scale_a = params_dict['scale_a']
    scale_b = params_dict['scale_b']
    e = 1e-6
    cm_recovered = cm + scale_a*fm + scale_b*cm/(gaussian_filter(fm, sigma=1)+e)
    cm_recovered = feature_engineering.normalize_matrix(cm_recovered)
    
    # RCW; Calculate from RCM
    cw_fitting = np.mean(cm_recovered, axis=0)
    cw_fitting = prune_cw(cw_fitting)
    return cw_fitting

# %% Optimization
def optimize_and_store(name, loss_fn, bounds, param_keys, distance_matrix, connectivity_matrix):
    res = differential_evolution(loss_fn, bounds=bounds, strategy='best1bin', maxiter=1000)
    params = dict(zip(param_keys, res.x))
    results[name] = {'params': params, 'loss': res.fun}
    fittings[name] = compute_cw_fitting(name, params, distance_matrix, connectivity_matrix)

def loss_fn_template(method_name, param_dict_fn, cw_target, distance_matrix, connectivity_matrix):
    def loss_fn(params):
        loss = np.mean((compute_cw_fitting(method_name, param_dict_fn(params), distance_matrix, connectivity_matrix) - cw_target) ** 2)
        return loss
    return loss_fn

# %% Usage
if __name__ == '__main__':
    results, fittings = {}, {}
    
    # Fittin target and DM
    manual_bad_channels = [57, 61]
    cw_target, distance_matrix, connectivity_matrix = prepare_target_and_inputs('PCC', 'label_driven_mi_origin', manual_bad_channels)

    # %% Validation of Fitting Comparison; Before Fitting
    # Electrode labels
    from utils import utils_feature_loading
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    if hasattr(electrodes, 'tolist'):
        electrodes = electrodes.tolist()

    for i in sorted(manual_bad_channels, reverse=True):
        del electrodes[i]
    x = electrodes
    # Electrode names 作为横轴
    
    # Load and normalize non-modeled r
    cm_global_averaged = connectivity_matrix.copy()
    cw_non_moldeled = np.mean(cm_global_averaged, axis=0).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
    cw_non_moldeled = scaler.fit_transform(cw_non_moldeled).flatten()
    
    # Compute MSE
    from sklearn.metrics import mean_squared_error
    mse_nonmodeled = mean_squared_error(cw_target, cw_non_moldeled)
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(x, cw_target, label='cw_target', linestyle='--', marker='o', color='black')
    plt.plot(x, cw_non_moldeled, label='cw_non_modeled (before)', marker='x', linestyle=':')
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
    optimize_and_store(
        'exponential',
        loss_fn_template('gaussian', lambda p: {'sigma': p[0], 'scale_a': p[1], 'scale_b': p[2]}, cw_target, distance_matrix, connectivity_matrix),
        [(0.1, 20.0), (-1.0, 1.0), (0.01, 2.0)],
        ['sigma', 'scale_a', 'scale_b'],
        distance_matrix, connectivity_matrix
    )
    
    optimize_and_store(
        'gaussian',
        loss_fn_template('gaussian', lambda p: {'sigma': p[0], 'scale_a': p[1], 'scale_b': p[2]}, cw_target, distance_matrix, connectivity_matrix),
        [(0.1, 20.0), (-1.0, 1.0), (0.01, 2.0)],
        ['sigma', 'scale_a', 'scale_b'],
        distance_matrix, connectivity_matrix
    )
    
    optimize_and_store(
        'inverse',
        loss_fn_template('inverse', lambda p: {'sigma': p[0], 'alpha': p[1], 'scale_a': p[2], 'scale_b': p[3]}, cw_target, distance_matrix, connectivity_matrix),
        [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
        ['sigma', 'alpha', 'scale_a', 'scale_b'],
        distance_matrix, connectivity_matrix
    )
    
    optimize_and_store(
        'powerlaw',
        loss_fn_template('powerlaw', lambda p: {'alpha': p[0], 'scale_a': p[1], 'scale_b': p[2]}, cw_target, distance_matrix, connectivity_matrix),
        [(0.1, 10.0), (0.01, 1.0), (-1.0, 2.0)],
        ['alpha', 'scale_a', 'scale_b'],
        distance_matrix, connectivity_matrix
    )
    
    optimize_and_store(
        'rational_quadratic',
        loss_fn_template('rational_quadratic', lambda p: {'sigma': p[0], 'alpha': p[1], 'scale_a': p[2], 'scale_b': p[3]}, cw_target, distance_matrix, connectivity_matrix),
        [(0.1, 20.0), (0.1, 10.0), (-1.0, 1.0), (0.01, 2.0)],
        ['sigma', 'alpha', 'scale_a', 'scale_b'],
        distance_matrix, connectivity_matrix
    )
    
    optimize_and_store(
        'generalized_gaussian',
        loss_fn_template('generalized_gaussian', lambda p: {'sigma': p[0], 'beta': p[1], 'scale_a': p[2], 'scale_b': p[3]}, cw_target, distance_matrix, connectivity_matrix),
        [(0.1, 20.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
        ['sigma', 'beta', 'scale_a', 'scale_b'],
        distance_matrix, connectivity_matrix
    )
    
    optimize_and_store(
        'sigmoid',
        loss_fn_template('sigmoid', lambda p: {'mu': p[0], 'beta': p[1], 'scale_a': p[2], 'scale_b': p[3]}, cw_target, distance_matrix, connectivity_matrix),
        [(0.1, 10.0), (0.1, 5.0), (-1.0, 1.0), (0.01, 2.0)],
        ['mu', 'beta', 'scale_a', 'scale_b'],
        distance_matrix, connectivity_matrix
    )

    print("=== Fitting Results of All Models (Minimum MSE) ===")
    for method, result in results.items():
        print(f"[{method.upper()}] Best Parameters: {result['params']}, Minimum MSE: {result['loss']:.6f}")

    # %% Validation of Fitting Comparison
    from utils import utils_feature_loading
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    if hasattr(electrodes, 'tolist'):
        electrodes = electrodes.tolist()
    
    for i in sorted(manual_bad_channels, reverse=True):
        del electrodes[i]
    x = electrodes.copy() # Electrode names 作为横轴
    
    for method, cw_fitting in fittings.items():
        plt.figure(figsize=(10, 4))  # 每张图单独设置大小
        plt.plot(x, cw_target, label='cw_target', linestyle='--', marker='o')
        plt.plot(x, cw_fitting, label=f'cw_fitting ({method})', marker='x')
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
    heatmap_data = np.vstack([cw_target] + [fittings[method] for method in fittings.keys()])
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
    for i in sorted(manual_bad_channels, reverse=True):
        del electrodes[i]
    
    # target
    cw_target_ = cw_target.copy()
    # Apply Spatial Gaussian smoothing to target
    coordinates = utils_feature_loading.read_distribution('seed')
    coordinates = coordinates.drop(index=manual_bad_channels)
    
    cw_target_smooth = spatial_gaussian_smoothing(cw_target_, coordinates, sigma=20.0)
    drawer_channel_weight.draw_2d_mapping(cw_target_smooth, coordinates, electrodes)
    
    # _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(cw_target_smooth, electrodes)
    # drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    # # non-fitted
    # cw_non_fitted = vce_modeling.load_global_averages(feature='PCC')
    # cw_non_fitted = np.mean(cw_non_fitted, axis=0)
    # _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(cw_non_fitted, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    # drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    # # fitted
    # cw_fitted_g_gaussian = fittings['generalized_gaussian']
    # _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(cw_fitted_g_gaussian, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    # drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    # cw_fitted_inverse = fittings['inverse']
    # _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(cw_fitted_inverse, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    # drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    # cw_fitted_sigmoid = fittings['sigmoid']
    # _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(cw_fitted_sigmoid, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    # drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])

    # # %% Resort Fittings; For Saving
    # # weights_sigmoid = fittings['sigmoid'].copy()
    # # from utils import utils_feature_loading
    # # electrodes = utils_feature_loading.read_distribution('seed')['channel']
    # # rank_sigmoid = {'weights': weights_sigmoid, 'electrodes': electrodes}
    # # # 获取排序后的索引（按权重降序）
    # # sorted_indices = np.argsort(-weights_sigmoid)  # 负号表示降序排序
    
    # # # 构造排序后的结果
    # # ranked_sigmoid = {
    # #     'weights': weights_sigmoid[sorted_indices],               # 排序后的权重
    # #     'electrodes': electrodes[sorted_indices],    # 排序后的电极
    # #     'original_indices': sorted_indices               # 排序后电极在原始 electrodes 中的索引
    # # }
    
    # # weights_g_gaussian = fittings['generalized_gaussian'].copy()
    # # from utils import utils_feature_loading
    # # electrodes = utils_feature_loading.read_distribution('seed')['channel']
    # # rank_g_gaussian = {'weights': weights_sigmoid, 'electrodes': electrodes}
    # # # 获取排序后的索引（按权重降序）
    # # sorted_indices = np.argsort(-weights_g_gaussian)  # 负号表示降序排序
    
    # # # 构造排序后的结果
    # # ranked_g_gaussian = {
    # #     'weights': weights_g_gaussian[sorted_indices],               # 排序后的权重
    # #     'electrodes': electrodes[sorted_indices],    # 排序后的电极
    # #     'original_indices': sorted_indices               # 排序后电极在原始 electrodes 中的索引
    # # }