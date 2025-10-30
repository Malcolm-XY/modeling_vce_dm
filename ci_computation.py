# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 23:43:14 2025

@author: 18307
"""

import numpy as np
import pandas as pd
import scipy.signal
from scipy.stats import f_oneway

from utils import utils_feature_loading, utils_visualization, utils_eeg_loading

# %% Preprocessing
def downsample_mean(data, factor):
    channels, points = data.shape
    truncated_length = points - (points % factor)  # 确保整除
    data_trimmed = data[:, :truncated_length]  # 截断到可整除的长度
    data_downsampled = data_trimmed.reshape(channels, -1, factor).mean(axis=2)  # 每 factor 组取平均值
    return data_downsampled

def downsample_decimate(data, factor):
    return scipy.signal.decimate(data, factor, axis=1, ftype='fir', zero_phase=True)

def up_sampling(data, factor):
    new_length = len(data) * factor
    data_upsampled = scipy.signal.resample(data, new_length)
    return data_upsampled

def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# %% Feature Computation
# Mutual Information
from dtaidistance import dtw
def compute_dtw(x, y):
    return dtw.distance(x, y)

from scipy.stats import pearsonr
def compute_pcc(x, y):
    r, _ = pearsonr(x, y)
    return r

from scipy.stats import spearmanr
def compute_sc(x, y):
    r, _ = spearmanr(x, y)
    return r

from scipy.signal import correlate
def compute_cc(x, y):
    """最大归一化交叉相关，忽略滞后信息"""
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    corr = correlate(x, y, mode='full')
    max_corr = np.max(np.abs(corr)) / len(x)
    return max_corr

def compute_mi(x, y):
    """ Fast mutual information computation using histogram method. """
    hist_2d, _, _ = np.histogram2d(x, y, bins=5)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    nonzero = pxy > 0  # Avoid log(0)
    return np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))

# A-NOVA
def compute_anova(x, y):
    f_stat, p_value = f_oneway(x, y)
    return f_stat, p_value

# %% Features Computation
def compute_feature_array(xs, y, method, electrodes=None, verbose=False, visualize=False):
    method = method.lower()
    SUPPORTED = ['mi', 'a-nova', 'pcc', 'sc', 'dtw', 'cc']
    if method not in SUPPORTED:
        raise ValueError(f"method must be one of {SUPPORTED}, got '{method}'")
        
    feature_array = []
    for idx, x in enumerate(xs):
        if verbose:
            print(f"Current idx: {idx}")
        if method == 'mi':
            feature = compute_mi(x, y)
        elif method == 'a-nova':
            f_stat, p_val = compute_anova(x, y)
            feature = f_stat
        elif method == 'pcc':
            feature = compute_pcc(x, y)
        elif method == 'sc':
            feature = compute_sc(x, y)
        elif method == 'dtw':
            feature = compute_dtw(x, y)
        elif method == 'cc':
            feature = compute_cc(x, y)
        else:
            feature = np.nan
        feature_array.append(feature)
    
    normalized_features = min_max_normalize(feature_array)
        
    if electrodes is not None:
        feature_array_df = pd.DataFrame({'electrodes': electrodes, method: feature_array})
        normalized_feature_array_df = pd.DataFrame({'electrodes': electrodes, method: normalized_features})
        
        if visualize: 
            feature_array_log = np.log(np.abs(feature_array_df[method]) + 1e-8)
            utils_visualization.draw_heatmap_1d(feature_array_log, electrodes)
        
        return feature_array_df, normalized_feature_array_df
    
    if visualize: 
        feature_array_log = np.log(np.abs(np.array(feature_array)) + 1e-8)
        utils_visualization.draw_heatmap_1d(feature_array_log, electrodes)
    
    return feature_array, normalized_features

def compute_feature_array_(xs, y, method, electrodes=None, verbose=False, visualize=False):
    method = method.lower()
    if method not in ['mi', 'a-nova']:
        raise ValueError(f"method must be one of ['mi', 'a-nova'], got '{method}'")
        
    feature_array = []
    for idx, x in enumerate(xs):
        if verbose:
            print(f"Current idx: {idx}")
        
        if method == 'mi':
            feature = compute_mi(x, y)
        elif method == 'a-nova':
            # 只取F值，当然你也可以用p值，看你的需求
            f_stat, p_val = compute_anova(x, y)
            feature = f_stat  # 或者 feature = p_val
            
        feature_array.append(feature)
    
    normalized_features = min_max_normalize(feature_array)
        
    if electrodes is not None:
        feature_array_df = pd.DataFrame({'electrodes': electrodes, method: feature_array})
        normalized_feature_array_df = pd.DataFrame({'electrodes': electrodes, method: normalized_features})
        
        if visualize: 
            feature_array_log = np.log(feature_array_df[method] + 1e-8)  # 防止log(0)
            utils_visualization.draw_heatmap_1d(feature_array_log, electrodes)
        
        return feature_array_df, normalized_feature_array_df
    
    if visualize: 
        feature_array_log = np.log(np.array(feature_array) + 1e-8)
        utils_visualization.draw_heatmap_1d(feature_array_log, electrodes)
    
    return feature_array, normalized_features

# %% Compute Feature Arrays; Specific for SEED
def Compute_Feature_Mean_SEED(subject_range=range(1,2), experiment_range=range(1,2), electrodes=None,
                              dataset='SEED', align_method='upsampling', method='mi', visualize=False):
    """
    计算SEED数据集某种特征（如MI、ANOVA）的平均值，支持多被试和实验。
    """
    # labels upsampling    
    labels = np.reshape(utils_feature_loading.read_labels(dataset, header=True), -1)
    
    feature_list = []
    normed_feature_list = []

    for subject in subject_range:
        for experiment in experiment_range:
            identifier = f'sub{subject}ex{experiment}'
            print(f"Current processing {identifier}")
            eeg_sample = utils_eeg_loading.read_and_parse_seed(identifier)  # shape: [channels, points]
            
            num_channels, len_points = eeg_sample.shape
            factor = int(len_points / len(labels))
            
            if align_method.lower() == 'upsampling':
                eeg_sample_transformed = eeg_sample
                labels_transformed = up_sampling(labels, factor=factor)
            elif align_method.lower() == 'downsampling':
                eeg_sample_transformed = downsample_decimate(eeg_sample, factor=factor)
                labels_transformed = labels
            else:
                raise ValueError(f"Unknown align_method: {align_method}")
            
            # 对齐
            min_length = min(eeg_sample_transformed.shape[1], len(labels_transformed))
            eeg_sample_alined = eeg_sample_transformed[:, :min_length]
            labels_alined = labels_transformed[:min_length]

            # 利用compute_feature_array泛化计算特征
            features, normed_features = compute_feature_array(
                xs=eeg_sample_alined, y=labels_alined, method=method, electrodes=electrodes
                )
            # features为DataFrame, normed_features为DataFrame（如果有electrodes），否则为list
            # 为了保持一致，全部用DataFrame拼接
            feature_list.append(np.array(features[method]) if hasattr(features, '__getitem__') and method in features else np.array(features))
            normed_feature_list.append(np.array(normed_features[method]) if hasattr(normed_features, '__getitem__') and method in normed_features else np.array(normed_features))

    # 计算均值
    feature_mean = np.mean(feature_list, axis=0)
    normed_feature_mean = np.mean(normed_feature_list, axis=0)
    df_feature_mean = pd.DataFrame({'electrodes': electrodes, f'{method}': feature_mean})
    df_normed_feature_mean = pd.DataFrame({'electrodes': electrodes, f'{method}': normed_feature_mean})
    
    return df_feature_mean, df_normed_feature_mean

# %% Sample usage
def sample_usage():
    # === Composition
    # original sample of labels and eeg data
    identifier_sample = 'sub1ex1'
    labels = utils_feature_loading.read_labels('seed')
    eeg_sample = utils_eeg_loading.read_and_parse_seed(identifier_sample)
    
    # compute factor
    num_channels, len_points = eeg_sample.shape
    len_labels, _ = labels.shape
    factor = int(len_points / len_labels)
    
    # upsampling; for labels
    labels_upsampled = up_sampling(labels, factor)
    
    # down sampling; for data
    eeg_downsampled_sample = up_sampling(eeg_sample, factor)
    
    # compute sample of specified feature
    # upsampling method
    eeg_sample_0 = eeg_sample[0]
    mi_sample_upmethod = compute_mi(eeg_sample_0, labels_upsampled)
    anova_sample_upmethod = compute_anova(eeg_sample_0, labels_upsampled)
    
    # downsampling method
    eeg_downsampled_sample_0 = eeg_downsampled_sample[0]
    mi_sample_downmethod = compute_mi(eeg_downsampled_sample_0, labels)
    anova_sample_downmethod = compute_anova(eeg_downsampled_sample_0, labels)
    
    # === Intergrated
    # get electrodes
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    
    subject_range, experiment_range = range(1,2), range(1,2)
    mis_mean, mis_mean_normed = Compute_Feature_Mean_SEED(subject_range, experiment_range, 
                                                          electrodes, 'seed', 'downsampling', 'mi', True)

    anova_mean, anova_mean_normed = Compute_Feature_Mean_SEED(subject_range, experiment_range, 
                                                          electrodes, 'seed', 'downsampling', 'a-nova', True)    
    
    results = {
        'factor': factor,
        'labels': labels,
        'labels_upsampled': labels_upsampled,
        'eeg_sample': eeg_sample,
        'eeg_downsampled_sample': eeg_downsampled_sample,
        'mi_sample_upmethod': mi_sample_upmethod,
        'mi_sample_downmethod': mi_sample_downmethod,
        'anova_sample_upmethod': anova_sample_upmethod,
        'anova_sample_downmethod': anova_sample_downmethod,
        'mis_mean': mis_mean,
        'anova_mean': anova_mean
        }
    
    return results
    
if __name__ == "__main__":
    # get electrodes
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    
    # compute feature arrays
    measurement = 'mi' # 'mi', 'a-nova', 'pcc', 'sc', 'cc'
    subject_range, experiment_range = range(1,6), range(1,4)
    feature_arrays_mean, feature_arrays_mean_normed = Compute_Feature_Mean_SEED(subject_range, experiment_range, 
                                                electrodes, 'seed', 'upsampling', measurement)
    utils_visualization.draw_heatmap_1d(feature_arrays_mean[measurement], feature_arrays_mean['electrodes'])
    
    # get ascending weights and indices
    fas_mean_resorted = feature_arrays_mean.sort_values(measurement, ascending=False)
    utils_visualization.draw_heatmap_1d(fas_mean_resorted[measurement], fas_mean_resorted['electrodes'])