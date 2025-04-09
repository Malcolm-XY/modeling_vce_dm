# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 23:07:14 2025

@author: 18307
"""

import os
import h5py
import numpy as np

import feature_engineering
from utils import utils_feature_loading
from utils import utils_visualization

def load_global_averages(file_path=None, feature=None, band='joint'):
    """
    读取 HDF5 文件中的 global_alpha_average, global_beta_average, global_gamma_average 和 global_joint_average 数据。

    Args:
        band: Select Frequency Band of EEG Data
        file_path (str, optional): HDF5 文件的完整路径。若为 None，则根据 feature 参数构造路径。
        feature (str, optional): 特征类型，如 'PCC'。仅当 file_path 为 None 时使用。

    Returns:
        tuple: 包含 global_alpha_average, global_beta_average, global_gamma_average 和 global_joint_average 的元组。
    """
    # 如果没有提供文件路径，则根据特征类型构造默认路径
    if file_path is None:
        if feature is None:
            file_path = 'Distribution/fc_global_averages.h5'
        else:
            path_current = os.getcwd()
            file_path = os.path.join(path_current, 'Distribution', 'Connectivity_Matrices_Averaged',
                                     feature.upper(), f'fc_global_averages_{feature}.h5')

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # 读取数据
    with h5py.File(file_path, 'r') as f:
        global_alpha_average = f['alpha'][:]
        global_beta_average = f['beta'][:]
        global_gamma_average = f['gamma'][:]
        global_delta_average = f['delta'][:]
        global_theta_average = f['theta'][:]
        global_joint_average = f['joint'][:]

    if band == 'joint':
        global_average = global_joint_average
    elif band == 'beta':
        global_average = global_beta_average
    elif band == 'gamma':
        global_average = global_gamma_average
    elif band == 'alpha':
        global_average = global_alpha_average
    elif band == 'delta':
        global_average = global_delta_average
    elif band == 'theta':
        global_average = global_theta_average
    else: raise NotImplementedError

    return global_average

def compute_volume_conduction_factors(_distance_matrix, method='exponential', params=None):
    """
    基于距离矩阵计算体积电导效应的因子矩阵。
    支持多种模型：exponential, gaussian, inverse, cutoff, powerlaw, rational_quadratic, generalized_gaussian, sigmoid

    Args:
        _distance_matrix (numpy.ndarray): 电极间的距离矩阵，形状为 (n, n)
        method (str): 建模方法
        params (dict): 模型参数字典

    Returns:
        numpy.ndarray: 因子矩阵，与 distance_matrix 同形状
    """
    import numpy as np

    _distance_matrix = np.asarray(_distance_matrix)

    # 默认参数集合
    default_params = {
        'exponential': {'sigma': 10.0},
        'gaussian': {'sigma': 5.0},
        'inverse': {'sigma': 5.0, 'alpha': 2.0},
        'cutoff': {'threshold': 5.0, 'factor': 0.5},
        'powerlaw': {'alpha': 2.0},
        'rational_quadratic': {'sigma': 5.0, 'alpha': 1.0},
        'generalized_gaussian': {'sigma': 5.0, 'beta': 2.0},
        'sigmoid': {'mu': 5.0, 'beta': 1.0},
    }

    if params is None:
        if method in default_params:
            params = default_params[method]
        else:
            raise ValueError(f"未提供参数，且方法 '{method}' 没有默认参数")
    elif method in default_params:
        method_params = default_params[method].copy()
        method_params.update(params)
        params = method_params
    else:
        raise ValueError(f"不支持的建模方法: {method}")

    # 初始化结果矩阵
    factor_matrix = np.zeros_like(_distance_matrix)
    epsilon = 1e-6  # 防止除0或log0

    if method == 'exponential':
        sigma = params['sigma']
        factor_matrix = np.exp(-_distance_matrix / sigma)

    elif method == 'gaussian':
        sigma = params['sigma']
        factor_matrix = np.exp(-np.square(_distance_matrix) / (sigma ** 2))

    elif method == 'inverse':
        sigma = params['sigma']
        alpha = params['alpha']
        factor_matrix = 1.0 / (1.0 + np.power(_distance_matrix / sigma, alpha))

    elif method == 'cutoff':
        threshold = params['threshold']
        factor = params['factor']
        factor_matrix = np.where(_distance_matrix < threshold, factor, 0.0)

    elif method == 'powerlaw':
        alpha = params['alpha']
        factor_matrix = 1.0 / (np.power(_distance_matrix, alpha) + epsilon)

    elif method == 'rational_quadratic':
        sigma = params['sigma']
        alpha = params['alpha']
        factor_matrix = np.power(1.0 + (np.square(_distance_matrix) / (2 * alpha * sigma ** 2)), -alpha)

    elif method == 'generalized_gaussian':
        sigma = params['sigma']
        beta = params['beta']
        factor_matrix = np.exp(-np.power(_distance_matrix / sigma, beta))

    elif method == 'sigmoid':
        mu = params['mu']
        beta = params['beta']
        factor_matrix = 1.0 / (1.0 + np.exp((_distance_matrix - mu) / beta))

    else:
        raise ValueError(f"不支持的体积电导建模方法: {method}")

    # 对角线置为1（自我连接）
    np.fill_diagonal(factor_matrix, 1.0)
    return factor_matrix

if __name__ == '__main__':  
    # %% Load Connectivity Matrix
    cm_pcc_joint = load_global_averages(feature='PCC')
    cm_pcc_joint = feature_engineering.normalize_matrix(np.array(cm_pcc_joint))
    utils_visualization.draw_projection(cm_pcc_joint)
    
    cm_pcc_joint = load_global_averages(feature='PLV')
    cm_pcc_joint = feature_engineering.normalize_matrix(np.array(cm_pcc_joint))
    utils_visualization.draw_projection(cm_pcc_joint)

    # %% Distance Matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix('seed')
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)
    utils_visualization.draw_projection(distance_matrix)

    _, distance_matrix_ste = feature_engineering.compute_distance_matrix('seed', method='stereo', stereo_params={'prominence': 0.1, 'epsilon': 0.01}, visualize=True)
    distance_matrix_ste = feature_engineering.normalize_matrix(distance_matrix_ste)
    utils_visualization.draw_projection(distance_matrix_ste)
    
    # %% Reversed Distance Matrix
    distance_matrix_r =  1 - distance_matrix
    utils_visualization.draw_projection(distance_matrix_r)
    
    distance_matrix_ste_r =  1 - distance_matrix_ste
    utils_visualization.draw_projection(distance_matrix_ste_r)
    
    # %% Similarity
    from sklearn.metrics.pairwise import cosine_similarity
    def cosine_sim(A, B):
        return cosine_similarity(A.flatten().reshape(1, -1), B.flatten().reshape(1, -1))[0][0]

    similarity_cosine_euclidean = cosine_sim(distance_matrix_r, cm_pcc_joint)
    print(f"The Cosine Similarity Between Euclidean Distance Matrix and Connectivity Matrix is: {similarity_cosine_euclidean}")
    similarity_cosine_stereo = cosine_sim(distance_matrix_ste_r, cm_pcc_joint)
    print(f"The Cosine Similarity Between Stereo Distance Matrix and Connectivity Matrix is: {similarity_cosine_stereo}")
    
    from skimage.metrics import structural_similarity as ssim
    similarity_ssim_euclidean = ssim(distance_matrix_r, cm_pcc_joint, data_range=1.0)
    print(f"The SSIM Similarity Between Euclidean Distance Matrix and Connectivity Matrix is: {similarity_ssim_euclidean}")
    similarity_ssim_ste = ssim(distance_matrix_ste_r, cm_pcc_joint, data_range=1.0)
    print(f"The SSIM Similarity Between Stereo Distance Matrix and Connectivity Matrix is: {similarity_ssim_ste}")

    def pearson_corr(A, B):
        return np.corrcoef(A.flatten(), B.flatten())[0, 1]
    similarity_corr_euclidean = pearson_corr(distance_matrix_r, cm_pcc_joint)
    print(f"The Correlation Similarity Between Euclidean Distance Matrix and Connectivity Matrix is: {similarity_corr_euclidean}")
    similarity_corr_ste = pearson_corr(distance_matrix_ste_r, cm_pcc_joint)
    print(f"The Correlation Similarity Between Stereo Distance Matrix and Connectivity Matrix is: {similarity_corr_ste}")
    
    # %% Factor Matrix
    factor_matrix = compute_volume_conduction_factors(distance_matrix)
    factor_matrix = feature_engineering.normalize_matrix(factor_matrix)
    utils_visualization.draw_projection(factor_matrix)
    
    factor_matrix_ste = compute_volume_conduction_factors(distance_matrix_ste)
    factor_matrix_ste = feature_engineering.normalize_matrix(factor_matrix_ste)
    utils_visualization.draw_projection(factor_matrix_ste)

    # %% Recovered Connectivity Matrix; Close to Genuine Connectivity Matrix
    differ_PCC_DM = cm_pcc_joint - factor_matrix
    utils_visualization.draw_projection(differ_PCC_DM)
    
    differ_PCC_DM_ste = cm_pcc_joint - factor_matrix_ste
    utils_visualization.draw_projection(differ_PCC_DM_ste)
    
    # %% Recovered Channel Weight
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    
    channel_weight = np.mean(differ_PCC_DM, axis=0)
    utils_visualization.draw_heatmap_1d(channel_weight, electrodes)
    
    channel_weight_ste = np.mean(differ_PCC_DM_ste, axis=0)
    utils_visualization.draw_heatmap_1d(channel_weight_ste, electrodes)
    
    # %% Label-Driven-MI-Based Channel Weight
    import drawer_channel_weight
    weights_LD_MI = drawer_channel_weight.get_ranking_weight(ranking='label_driven_mi')
    index = drawer_channel_weight.get_index(ranking='label_driven_mi')
    utils_visualization.draw_heatmap_1d(weights_LD_MI, electrodes)
    drawer_channel_weight.draw_weight_map_from_data(index, weights_LD_MI)
    
    # %% Matrix of differ(Connectivity_Matrix_PCC, Factor_Matrix); stereo distance matrix; generalized_gaussian
    # Target
    import drawer_channel_weight
    drawer_channel_weight.draw_weight_map_from_file(ranking_method='label_driven_mi')
    
    # Fitted
    channel_names, distance_matrix = feature_engineering.compute_distance_matrix('seed', method='stereo')
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)
    utils_visualization.draw_projection(distance_matrix)

    factor_matrix = compute_volume_conduction_factors(distance_matrix, method='generalized_gaussian', params={'sigma': 2.27, 'beta': 5.0})
    factor_matrix = feature_engineering.normalize_matrix(factor_matrix)
    utils_visualization.draw_projection(factor_matrix)

    global_joint_average = load_global_averages(feature='PCC')
    global_joint_average = feature_engineering.normalize_matrix(global_joint_average)
    utils_visualization.draw_projection(global_joint_average)

    differ_PCC_DM = global_joint_average - factor_matrix
    utils_visualization.draw_projection(differ_PCC_DM)
    
    # transform from Matrix to Rank
    weight_fitted = np.mean(differ_PCC_DM, axis=0)
    from sklearn.preprocessing import MinMaxScaler
    weight_fitted = MinMaxScaler().fit_transform(weight_fitted.reshape(-1, 1)).flatten()
    from scipy.stats import boxcox
    weight_fitted = weight_fitted + 1e-6
    weight_fitted, _ = boxcox(weight_fitted)
    
    # Visualiztion
    # get electrodes
    distribution = utils_feature_loading.read_distribution('seed')
    electrodes = distribution['channel']
    
    # resort
    weight_channels = np.mean(differ_PCC_DM, axis=0)
    strength_origin, strength_ranked, rank_indices = drawer_channel_weight.rank_channel_strength(weight_fitted, electrodes) #, exclude_electrodes=['CB1', 'CB2'])
    
    drawer_channel_weight.draw_weight_map_from_data(rank_indices, strength_ranked['Strength'])