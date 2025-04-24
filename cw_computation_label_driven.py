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

# %% Feature Computation
# Mutual Information
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
    """
    Computes one-way ANOVA (analysis of variance) between two groups: x and y.

    Parameters:
    x (array-like): First sample group.
    y (array-like): Second sample group.

    Returns:
    f_stat (float): The computed F-statistic.
    p_value (float): The associated p-value.
    """
    f_stat, p_value = f_oneway(x, y)
    return f_stat, p_value

# %% Features Computation
def compute_feature_array(xs, y, method, electrodes=None, verbose=True):
    method = method.lower()
    if method not in ['mi', 'a-nova']:
        raise ValueError(f"method must be one of ['mi', 'a-nova'], got '{method}'")
        
    feature_array = []
    for x in xs:
        if verbose:
            print(f"For x in computing feature array: {x.shape}")
            print(f"For y in computing feature array: {y.shape}")
        
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
        
        if verbose: 
            feature_array_log = np.log(feature_array_df[method] + 1e-8)  # 防止log(0)
            utils_visualization.draw_heatmap_1d(feature_array_log, electrodes)
        
        return feature_array_df, normalized_feature_array_df
    
    if verbose: 
        feature_array_log = np.log(np.array(feature_array) + 1e-8)
        utils_visualization.draw_heatmap_1d(feature_array_log, electrodes)
    
    return feature_array, normalized_features
        
def compute_mis(xs, y, electrodes=None, verbose=True):
    mis = []
    for x in xs:
        print(f"For x in computing mis: {x.shape}")
        print(f"For y in computing mis: {y.shape}")
        mi = compute_mi(x, y)
        mis.append(mi)
    
    mis = np.array(mis,  dtype=float)
    normalized_mis = min_max_normalize(mis)

    if electrodes is not None:
        mis = pd.DataFrame({'electrodes': electrodes, 'mis': mis})
        
        normalized_mis = pd.DataFrame({'electrodes': electrodes, 'mis': normalized_mis})
        
        if verbose: 
            mis_log = np.log(mis['mis'])
            utils_visualization.draw_heatmap_1d(mis_log, electrodes)
        
        return mis, normalized_mis
    
    if verbose: 
        mis_log = np.log(mis)
        utils_visualization.draw_heatmap_1d(mis_log, electrodes)
    
    return mis, normalized_mis
    
def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# %% Compute MIs; Specific for SEED
def Compute_MIs_Mean_SEED(subject_range=range(1,2), experiment_range=range(1,2), electrodes=None,
                     dataset='SEED', align_method='upsampling', verbose=False):
    # labels upsampling    
    labels = np.reshape(utils_feature_loading.read_labels(dataset), -1)
    
    # compute mis_mean
    mis, mis_normalized = [], []
    subject_range, experiment_range = subject_range, experiment_range
    for subject in subject_range:
        for experiment in experiment_range:
            
            identifier = f'sub{subject}ex{experiment}'
            
            eeg_sample = utils_eeg_loading.read_and_parse_seed(identifier)
            
            # transform two variables by up/down sampling
            num_channels, len_points = eeg_sample.shape
            factor=int(len_points/len(labels))
            
            if align_method.lower() == 'upsampling':
                eeg_sample_transformed = eeg_sample
                labels_transformed = up_sampling(labels, factor=factor)
                
            elif align_method.lower() == 'downsampling':
                eeg_sample_transformed = downsample_decimate(eeg_sample, factor=factor)
                labels_transformed = labels
            
            # alin two variables
            min_length = min(eeg_sample_transformed.shape[1], len(labels_transformed))
            eeg_sample_alined = eeg_sample_transformed[:, :min_length]
            labels_alined = labels_transformed[:min_length]

            # compute mis
            mis_temp, mis_sampled_temp = compute_mis(eeg_sample_alined, labels_alined, electrodes=electrodes)
            
            mis.append(mis_temp)
            mis_normalized.append(mis_normalized)
    
    # arrange mis_mean
    mis_mean = np.array(np.array(mis)[:,:,1].mean(axis=0), dtype=float)
    mis_mean_ = pd.DataFrame({'electrodes':electrodes, 'mi_mean':mis_mean})
    
    # plot heatmap
    utils_visualization.draw_heatmap_1d(mis_mean, electrodes)
    utils_visualization.draw_heatmap_1d(np.log(mis_mean), electrodes)
    
    # get ascending indices
    mis_mean_resorted = mis_mean_.sort_values('mi_mean', ascending=False)
    utils_visualization.draw_heatmap_1d(np.log(mis_mean_resorted['mi_mean']), mis_mean_resorted['electrodes'])
    
    return mis_mean_, mis_mean_resorted

if __name__ == "__main__":
    # get electrodes
    distribution = utils_feature_loading.read_distribution('seed')
    electrodes = distribution['channel']
    
    # labels upsampling
    labels = utils_feature_loading.read_labels('seed')
    labels_upsampled = up_sampling(labels, 200)
    
    # compute mis_mean
    subject_range, experiment_range = range(1,16), range(1,4)
    mis_mean_, mis_mean_resorted = Compute_MIs_Mean_SEED(subject_range, experiment_range, electrodes, align_method='upsampling', verbose=False)
