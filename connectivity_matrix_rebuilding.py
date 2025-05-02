# -*- coding: utf-8 -*-
"""
Created on Fri May  2 23:50:20 2025

@author: 18307
"""

from vce_modeling import compute_volume_conduction_factors_basic_model as compute_fm_basic
from vce_modeling import compute_volume_conduction_factors_advanced_model as compute_fm_advanced
import vce_model_fitting

def cm_rebuilding(cms, distance_matrix, params, model='exponential', model_fm='basic', model_rcm='differ'):
    if model not in ['exponential', 'gaussian', 'inverse', 'generalized_gaussian', 'powerlaw', 'rational_quadratic', 'sigmoid']:
        raise ValueError("Chosen model is not supported.")
    if model_fm not in ['basic', 'advanced']:
        raise ValueError("Model_FM must be chosen as 'Basic' or 'Advanced'.")
    if model_rcm not in ['differ', 'linear', 'linear_ratio']:
        raise ValueError("Model_RCM must be chosen as 'differ', 'linear' or 'linear_ratio'.")
    
    scale_a = params.get('scale_a', 0)
    scale_b = params.get('scale_b', 0)
    
    if model_fm == 'basic':
        factor_matrix = compute_fm_basic(distance_matrix, model, params)
    elif model_fm == 'advanced':
        factor_matrix = compute_fm_advanced(distance_matrix, model, params)
        
    if model_rcm == 'differ':
        cm_rebuiled = cms - factor_matrix
    elif model_rcm == 'linear':
        cm_rebuiled = cms + scale_a * factor_matrix
    elif model_rcm == 'linear_ratio':
        e = 1e-6
        cm_rebuiled = cms + scale_a * factor_matrix + scale_b * cms / (vce_model_fitting.gaussian_filter(factor_matrix, sigma=1) + e)
        
    return cm_rebuiled

def example_usage():
    import numpy as np
    import feature_engineering
    from utils import utils_feature_loading, utils_visualization
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d"})
    dm = feature_engineering.normalize_matrix(dm)
    utils_visualization.draw_projection(dm, 'Distance Matrix')
    
    cms_sample = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')
    cm_sample = cms_sample.get('alpha', '')
    utils_visualization.draw_projection(np.mean(cm_sample, axis=0), 'Connectivity Matrix Sample')

    params = {'sigma': 0.2}
    model, model_fm, model_rcm = 'exponential', 'basic', 'differ'
    
    rcm = cm_rebuilding(cm_sample, dm, params, model, model_fm, model_rcm)
    utils_visualization.draw_projection(np.mean(rcm, axis=0), 'Rebuilded Connectivity Matrix Sample')

if __name__ == '__main__':
    example_usage()