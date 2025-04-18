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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

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

def clean_and_smooth_cm(CM, coordinates, param, return_mask=False):
    """
    Remove abnormal connections and fill them using spatial Gaussian smoothing (only on removed ones).

    Parameters
    ----------
    CM : np.ndarray (N, N)
        Symmetric connectivity matrix.
    coordinates : dict
        Dict with 'x', 'y', 'z' coordinate arrays of length N.
    param : dict
        - 'asym_thresh': float
        - 'sim_thresh': float
        - 'sigma': float
    return_mask : bool
        Whether to return the abnormal mask.

    Returns
    -------
    CM_cleaned : np.ndarray
        Cleaned and partially smoothed matrix.
    (optional) abnormal_mask : np.ndarray
        Boolean mask of removed connections.
    """
    N = CM.shape[0]
    A = CM.copy()

    # Step 1: Asymmetry detection
    asym_map = np.abs(A - A.T)
    asym_mask = (asym_map > param['asym_thresh'])

    # Step 2: Spatial similarity
    conn_features = A
    row_sim = cosine_similarity(conn_features)
    coords = np.vstack([coordinates['x'], coordinates['y'], coordinates['z']]).T
    dists = cdist(coords, coords)
    dist_thresh = np.percentile(dists, 10)
    sim_mask = np.zeros_like(A, dtype=bool)

    for i in range(N):
        for j in range(N):
            if i != j and dists[i, j] < dist_thresh and row_sim[i, j] < param['sim_thresh']:
                sim_mask[i, j] = True

    # Step 3: Combine masks
    abnormal_mask = np.logical_or(asym_mask, sim_mask)

    # Step 4: Remove abnormal values
    A[abnormal_mask] = 0
    A = 0.5 * (A + A.T)

    # Step 5: Fill only removed entries using spatial Gaussian smoothing
    weights = np.exp(- (dists ** 2) / (2 * param['sigma'] ** 2))
    weights /= weights.sum(axis=1, keepdims=True)
    A_smooth = weights @ A @ weights.T

    A_filled = A.copy()
    A_filled[abnormal_mask] = A_smooth[abnormal_mask]
    A_filled = 0.5 * (A_filled + A_filled.T)

    if return_mask:
        return A_filled, abnormal_mask
    else:
        return A_filled

def remove_and_fill_channels_knn(CM, coordinates, param):
    """
    Detect abnormal EEG channels from connectivity matrix and fill using KNN-based imputation.

    Parameters:
    -----------
    CM : ndarray
        Functional connectivity matrix (n_channels x n_channels)
    coordinates : dict
        Dictionary with keys 'x', 'y', 'z', each of shape (n_channels,)
    param : dict
        Parameters including:
            - 'threshold': float, e.g. 2.5 (Z-score or IQR threshold)
            - 'method': str, either 'zscore' or 'iqr'
            - 'k': int, number of nearest neighbors to use for imputation

    Returns:
    --------
    CM_filled : ndarray
        Filled connectivity matrix after bad channels removed and reconstructed.
    """
    n_channels = CM.shape[0]
    CM_filled = CM.copy()

    # === 1. Get 3D coordinates
    coords = np.vstack([coordinates['x'], coordinates['y'], coordinates['z']]).T

    # === 2. Detect bad channels based on mean abs connectivity
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

    print(f"[KNN Fill] Detected bad channels: {bad_idx.tolist()}")

    if len(bad_idx) == 0:
        return CM_filled  # No bad channels, return original

    # === 3. Fit Nearest Neighbors
    k = param.get('k', 5)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)  # +1 to exclude self

    for i in bad_idx:
        dists, indices = nbrs.kneighbors([coords[i]])
        neighbors = [j for j in indices[0] if j != i][:k]

        # Replace row and column using mean of neighbors
        CM_filled[i, :] = np.mean(CM[neighbors, :], axis=0)
        CM_filled[:, i] = np.mean(CM[:, neighbors], axis=1)
        CM_filled[i, i] = 0  # Clean diagonal

    return CM_filled

def preprocessing_global_averaged_cm(feature='PCC'):
    # Global averaged connectivity matrix; For subsquent fitting computation
    global_joint_average = vce_modeling.load_global_averages(feature=feature)
    global_joint_average = np.abs(global_joint_average)
    connectivity_matrix = feature_engineering.normalize_matrix(global_joint_average)
    
    # Replace bad channel; F7
    # connectivity_matrix[5, :] = connectivity_matrix[6, :].copy()
    # connectivity_matrix[:, 5] = connectivity_matrix[:, 6].copy()
    
    # connectivity_matrix[27, :] = np.mean(connectivity_matrix, axis=0).copy()
    # connectivity_matrix[:, 27] = np.mean(connectivity_matrix, axis=1).copy()
    
    # Apply Spatial Gaussian Smoothing to CM
    # connectivity_matrix = gaussian_filter(connectivity_matrix, sigma=0.5)
    
    # Apply Spatial Gaussian Smoothing for CM to CM
    from utils import utils_feature_loading, utils_visualization
    utils_visualization.draw_projection(connectivity_matrix)
    
    coordinates = utils_feature_loading.read_distribution('seed')
    param = {'threshold': 1.5, 'method': 'zscore', 'k': 5}
    connectivity_matrix = remove_and_fill_channels_knn(connectivity_matrix, coordinates, param)
    
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

def prepare_target_and_inputs(feature='PCC', ranking_method='label_driven_mi_origin'):
    # Target; Label-Driven CW
    weights = drawer_channel_weight.get_ranking_weight('label_driven_mi_origin')
    cw_target = prune_cw(weights.to_numpy())
    
    # Apply Spatial Gaussian smoothing to target
    from utils import utils_feature_loading
    coordinates = utils_feature_loading.read_distribution('seed')
    cw_target_smooth = spatial_gaussian_smoothing(cw_target, coordinates, sigma=20.0)
    
    # Distance matrix; For subsquent fitting computation
    _, distance_matrix = feature_engineering.compute_distance_matrix('seed',
        projection_params={'type': 'azimuthal', 'y_compression_factor': 0.5,'y_compression_direction': 'negative'})
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
    cw_target, distance_matrix, connectivity_matrix = prepare_target_and_inputs('PCC', 'label_driven_mi_origin')

    # %% Validation of Fitting Comparison; Before Fitting
    # Electrode labels
    from utils import utils_feature_loading
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    if hasattr(electrodes, 'tolist'):
        electrodes = electrodes.tolist()
    
    x = electrodes  # Electrode names 作为横轴
    
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
    
    x = electrodes  # Electrode names 作为横轴
    
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
    
    # target
    cw_target_ = cw_target.copy()
    # Apply Spatial Gaussian smoothing to target
    coordinates = utils_feature_loading.read_distribution('seed')
    cw_target_smooth = spatial_gaussian_smoothing(cw_target_, coordinates, sigma=20.0)
    _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(cw_target_smooth, electrodes)
    drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    # non-fitted
    cw_non_fitted = vce_modeling.load_global_averages(feature='PCC')
    cw_non_fitted = np.mean(cw_non_fitted, axis=0)
    _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(cw_non_fitted, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    # fitted
    cw_fitted_g_gaussian = fittings['generalized_gaussian']
    _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(cw_fitted_g_gaussian, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    cw_fitted_inverse = fittings['inverse']
    _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(cw_fitted_inverse, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])
    
    cw_fitted_sigmoid = fittings['sigmoid']
    _, strength_ranked, in_original_indices = drawer_channel_weight.rank_channel_strength(cw_fitted_sigmoid, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    drawer_channel_weight.draw_weight_map_from_data(in_original_indices, strength_ranked['Strength'])

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
        'weights': weights_g_gaussian[sorted_indices],               # 排序后的权重
        'electrodes': electrodes[sorted_indices],    # 排序后的电极
        'original_indices': sorted_indices               # 排序后电极在原始 electrodes 中的索引
    }