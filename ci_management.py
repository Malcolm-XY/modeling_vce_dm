# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:43:47 2025

@author: 18307
"""
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import utils_feature_loading

def read_channel_importances(folder='channel_importances', excel='channel_importances_LD.xlsx', sheet='label_driven_mi_1_5', sort=False):
    # define path
    path_current = os.getcwd()

    path_excel = os.path.join(path_current, folder, excel)
    
    # read xlxs; channel importance    
    channel_importances = pd.read_excel(path_excel, sheet_name=sheet, engine='openpyxl')
    importances = channel_importances[['labels','ams']]
    
    if sort:
        importances = importances.sort_values(by='ams', ascending=False)

    return importances

def read_channel_importances_fitted(model_fm='basic', model_rcm='differ', model='exponential', 
                                source='fitted_results(sub1_sub5_joint_band)', sort=False):
    model_fm = model_fm.lower()
    model_rcm = model_rcm.lower()
    model = model.lower()
    
    path_current = os.getcwd()
    path_file = os.path.join(path_current, 'fitted_results', source, 
                             f'channel_importances({model_fm}_fm_{model_rcm}_rcm).xlsx')

    channel_importances = pd.read_excel(path_file, sheet_name=model, engine='openpyxl')
    
    importances = channel_importances[['labels','ams']]
    
    if sort:
        importances = importances.sort_values(by='ams', ascending=False)
    
    return importances

def rank_channel_importances(node_strengths, electrode_labels, ascending=False, exclude_electrodes=None):
    """
    Sort and visualize node strengths in a functional connectivity (FC) network,
    with optional electrode exclusion after sorting.

    Args:
        node_strengths (numpy.ndarray): 1D array of node strengths (e.g., mean connection strength per electrode).
        electrode_labels (list of str): List of electrode names corresponding to nodes.
        feature_name (str, optional): Name of the feature (used in plot title). Default is 'feature'.
        ascending (bool, optional): Sort order. True for ascending, False for descending. Default is False.
        draw (bool, optional): Whether to draw the heatmap. Default is True.
        exclude_electrodes (list of str, optional): List of electrode names to exclude *after* sorting.

    Returns:
        tuple:
            - df_original (pd.DataFrame): DataFrame sorted by strength, with index being sorted indices.
            - df_ranked (pd.DataFrame): DataFrame sorted by strength, with column 'OriginalIndex' showing original position.
            - sorted_indices (np.ndarray): Sorted indices (after exclusion) relative to the original list.
    """
    if len(electrode_labels) != len(node_strengths):
        raise ValueError(
            f"Length mismatch: {len(electrode_labels)} electrode labels vs {len(node_strengths)} strengths.")

    electrode_labels = list(electrode_labels)

    # Create full unsorted DataFrame
    df_unsorted = pd.DataFrame({
        'Electrode': electrode_labels,
        'Strength': node_strengths,
    })

    df_original = pd.DataFrame({
        'OriginalIndex': df_unsorted.index,
        'Electrode': electrode_labels,
        'Strength': node_strengths,
    })

    # Perform sorting
    sorted_df = df_unsorted.sort_values(by='Strength', ascending=ascending).reset_index()

    # sorted_df.index → sorted rank
    # sorted_df['index'] → original index
    sorted_df.rename(columns={'index': 'OriginalIndex'}, inplace=True)

    # Optional exclusion
    if exclude_electrodes is not None:
        df_ranked = sorted_df[~sorted_df['Electrode'].isin(exclude_electrodes)].reset_index(drop=True)
    else:
        df_ranked = sorted_df.copy()

    # Sorted indices (for matrix reordering)
    sorted_indices = df_ranked['OriginalIndex'].values

    return df_original, df_ranked, sorted_indices

def draw_importance_map_from_file(ranking_method='label_driven_mi_1_5', offset=0, transformation='log', reverse=False):
    # 获取数据
    importances = read_channel_importances(sheet=ranking_method)['ams']
    if reverse:
        importances = 1 - importances
    distribution = utils_feature_loading.read_distribution('seed')

    x = np.array(distribution['x'])
    y = np.array(distribution['y'])
    electrodes = distribution['channel']

    # 归一化 label_driven_mi_mean 以适应颜色显示（假设它是数值列表）
    if transformation == 'log':
        values = np.array(np.log(importances) + offset)
    else:
        values = np.array(importances + offset)

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=values, cmap='coolwarm', s=100, edgecolors='k')

    # 添加颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('Label Driven MI Mean')

    # 标注电极通道名称
    for i, txt in enumerate(electrodes):
        plt.text(x[i], y[i], txt, fontsize=9, ha='right', va='bottom')

    # 设置标题和坐标轴
    plt.title("Weight Distribution on Electrodes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()

    return None

def draw_importance_map_from_data(importances, offset=0, transformation=None, reverse=False):
    if reverse:
        importances = 1 - importances
    distribution = utils_feature_loading.read_distribution('seed')

    x = np.array(distribution['x'])
    y = np.array(distribution['y'])
    electrodes = distribution['channel']

    # 归一化 label_driven_mi_mean 以适应颜色显示（假设它是数值列表）
    if transformation == 'log':
        values = np.array(np.log(importances) + offset)
    else:
        values = np.array(importances + offset)

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=values, cmap='coolwarm', s=100, edgecolors='k')

    # 添加颜色条
    cbar = plt.colorbar(sc)
    cbar.set_label('Label Driven MI Mean')

    # 标注电极通道名称
    for i, txt in enumerate(electrodes):
        plt.text(x[i], y[i], txt, fontsize=9, ha='right', va='bottom')

    # 设置标题和坐标轴
    plt.title("Weight Distribution on Electrodes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()

    return None

if __name__ == '__main__':
    # miportances of target and fitted
    importances_target = read_channel_importances(sheet='label_driven_mi_1_5', sort=False)
    draw_importance_map_from_data(importances_target['ams'])
    importances_fitted = read_channel_importances_fitted(model_fm='basic', model_rcm='differ', model='exponential',
                                                 source='fitted_results(1_5_joint_band_from_mat)', sort=False)
    draw_importance_map_from_data(importances_fitted['ams'])
    
    # channel importance map
    draw_importance_map_from_file(transformation=None, ranking_method='label_driven_mi_1_5')
    draw_importance_map_from_file(transformation=None, ranking_method='label_driven_mi_10_15')
    
    channel_importances = read_channel_importances(sheet='label_driven_mi_1_5')['ams']
    draw_importance_map_from_data(channel_importances)
    channel_importances = read_channel_importances(sheet='label_driven_mi_10_15')['ams']
    draw_importance_map_from_data(channel_importances)
    
    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    df_original, df_ranked, sorted_indices = rank_channel_importances(channel_importances, electrodes)