# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 23:43:14 2025

@author: 18307
"""

import numpy as np
import pandas as pd
import scipy.signal

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

# %% Compute MIs
def compute_mi(x, y):
    """ Fast mutual information computation using histogram method. """
    hist_2d, _, _ = np.histogram2d(x, y, bins=5)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = np.outer(px, py)
    nonzero = pxy > 0  # Avoid log(0)
    return np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))

def compute_mis(xs, y, electrodes=None, assemble_electrodes=True, verbose=True):
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
    subject_range, experiment_range = range(1,2), range(1,4)
    mis_mean_, mis_mean_resorted = Compute_MIs_Mean_SEED(subject_range, experiment_range, electrodes, align_method='upsampling', verbose=False)
