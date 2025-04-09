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

def get_ranking_weight(ranking='label_driven_mi'):
    # define path
    path_current = os.getcwd()

    path_ranking = os.path.join(path_current, 'Distribution', 'Electrode_Rankings', 'electrodes_ranking.xlsx')

    # read xlxs; electrodes ranking
    ranking = pd.read_excel(path_ranking, sheet_name=ranking, engine='openpyxl')
    weight = ranking['mean']

    return weight

def get_index(ranking='label_driven_mi'):
    # define path
    path_current = os.getcwd()

    path_ranking = os.path.join(path_current, 'Distribution', 'Electrode_Rankings', 'electrodes_ranking.xlsx')

    # read xlxs; electrodes ranking
    ranking = pd.read_excel(path_ranking, sheet_name=ranking, engine='openpyxl')
    index = ranking['index(in origin dataset)']

    return index

def rank_channel_strength(node_strengths, electrode_labels, ascending=False, exclude_electrodes=None):
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

def draw_weight_map_from_file(ranking_method='label_driven_mi', offset=0, transformation='log', reverse=False):
    # 获取数据
    index = get_index(ranking_method)
    weights = get_ranking_weight(ranking_method)  # 假设它返回一个与 electrodes 对应的值列表
    if reverse:
        weights = 1 - weights
    distribution = utils_feature_loading.read_distribution('seed')

    dis_t = distribution.iloc[index]

    x = np.array(dis_t['x'])
    y = np.array(dis_t['y'])
    electrodes = dis_t['channel']

    # 归一化 label_driven_mi_mean 以适应颜色显示（假设它是数值列表）
    if transformation == 'log':
        values = np.array(np.log(weights) + offset)
    else:
        values = np.array(weights + offset)

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

def draw_weight_map_from_data(index, weights, ranked_electrodes=None, offset=0, transformation=None,
                              reverse=False):
    if reverse:
        weights = 1 - weights
    distribution = utils_feature_loading.read_distribution('seed')

    dis_t = distribution.iloc[index]

    x = np.array(dis_t['x'])
    y = np.array(dis_t['y'])
    electrodes = dis_t['channel']

    # 归一化 label_driven_mi_mean 以适应颜色显示（假设它是数值列表）
    if transformation == 'log':
        values = np.array(np.log(weights) + offset)
    else:
        values = np.array(weights + offset)

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
    # label-driven-MI
    weights, index = draw_weight_map_from_file(ranking_method='label_driven_mi')

    electrodes = utils_feature_loading.read_distribution('seed')['channel']
    weight_mean_r = weights[index]
    from utils import utils_visualization

    utils_visualization.draw_heatmap_1d(weight_mean_r, electrodes)

    # data-driven-MI
    draw_weight_map_from_file(transformation=None, ranking_method='data_driven_mi')

    draw_weight_map_from_file(transformation=None, ranking_method='data_driven_pcc')

    draw_weight_map_from_file(transformation=None, ranking_method='data_driven_plv')
