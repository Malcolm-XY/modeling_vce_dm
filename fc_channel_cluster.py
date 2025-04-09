# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:18:41 2025

@author: usouu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.spatial import ConvexHull

import utils
import fc_computer as fc

def hierarchical_clustering(distance_matrix, threshold=None, parse=False, verbose=False):
    """
    Perform hierarchical clustering to group signals based on a correlation matrix.

    :param correlation_matrix: ndarray
        Correlation coefficient matrix (n x n).
    :param threshold: float or None
        Dissimilarity threshold for clustering. If None, an automatic threshold is calculated 
        based on the average dissimilarity (default: None).
    :param parse: bool
        If True, parse the clusters into lists of grouped indices.
    :param verbose: bool
        If True, print additional information such as the number of clusters (default: False).
    :return: 
        clusters: ndarray
            Cluster labels for each signal.
        parsed_clusters: list (optional)
            Parsed clusters as groups of indices (if `parse=True`).
    """
    # Compute the distance matrix
    np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0 (self-distance)

    # Convert to condensed distance matrix for linkage
    condensed_dist = squareform(distance_matrix, checks=False)

    # Automatically determine the threshold if not provided
    if threshold is None:
        threshold = np.mean(condensed_dist)  # Use the mean of the condensed distance matrix
        if verbose:
            print(f"Automatically determined threshold: {threshold:.4f}")

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')

    # Parse clusters into groups of indices if required
    parsed_clusters = None
    if parse:
        parsed_clusters = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in parsed_clusters:
                parsed_clusters[cluster_id] = []
            parsed_clusters[cluster_id].append(idx)
        parsed_clusters = list(parsed_clusters.values())

        # Optionally print the number of clusters
        if verbose:
            print(f"The number of clusters: {len(parsed_clusters)}")

    return (clusters, parsed_clusters) if parse else clusters

def parse_clusters(cluster_labels):
    """
    将聚类标签解析为信号分组
    :param cluster_labels: 聚类标签列表
    :return: group_dict: 按标签分组的信号索引
    """
    group_dict = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        group_dict[label].append(idx)
        
        print(f"Cluster {label}: Index {idx}")
    return group_dict

def plot_3d_channels(distribution, clusters):
    """
    绘制 3D 通道分布，并按组用颜色区分。
    :param distribution: 包含 x, y, z 坐标和分组信息的 DataFrame
    """
    distribution['group'] = clusters
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 获取唯一组标签
    unique_groups = distribution['group'].unique()

    colormap = plt.colormaps['viridis']  # 获取调色盘
    colors = [colormap(i / len(unique_groups)) for i in range(len(unique_groups))]  # 动态分配颜色
    
    for idx, group in enumerate(unique_groups):
        group_data = distribution[distribution['group'] == group]
        ax.scatter(
            group_data['x'], group_data['y'], group_data['z'],
            label=f"Group {group}",
            color=colors[idx],  # 通过索引访问颜色
            s=50  # 点的大小
            )
        # 添加文本标签
        for _, row in group_data.iterrows():
            ax.text(row['x'], row['y'], row['z'], row['channel'], fontsize=8)

    ax.set_title("3D Channel Distribution with Groups")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

def plot_3d_channels_(distribution, clusters):
    """
    绘制 3D 通道分布，并按组用颜色区分，同时圈出相同组的点。
    :param distribution: 包含 x, y, z 坐标和分组信息的 DataFrame
    """
    distribution['group'] = clusters
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 获取唯一组标签
    unique_groups = distribution['group'].unique()
    colormap = plt.colormaps['tab10']  # 获取调色盘
    colors = [colormap(i / len(unique_groups)) for i in range(len(unique_groups))]  # 动态分配颜色

    for idx, group in enumerate(unique_groups):
        group_data = distribution[distribution['group'] == group]
        # 绘制组内的点
        ax.scatter(
            group_data['x'], group_data['y'], group_data['z'],
            label=f"Group {group}",
            color=colors[idx],
            s=50  # 点的大小
        )
        # 添加文本标签
        for _, row in group_data.iterrows():
            ax.text(row['x'], row['y'], row['z'], row['channel'], fontsize=8)

        # 绘制组的边界（凸包）
        if len(group_data) >= 4:  # 凸包要求至少 4 个点
            points = group_data[['x', 'y', 'z']].to_numpy()
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot_trisurf(
                    points[:, 0], points[:, 1], points[:, 2],
                    triangles=[simplex],
                    color=colors[idx],
                    alpha=0.2,  # 透明度
                    edgecolor='gray'
                )

    ax.set_title("3D Channel Distribution with Groups")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Load global averages for different frequency bands
    global_avg_alpha, global_avg_beta, global_avg_gamma = fc.load_global_averages()
    
    utils.draw_projection(global_avg_alpha)
    utils.draw_projection(global_avg_beta)
    utils.draw_projection(global_avg_gamma)
    
    # transform to distance matrix
    distance_matrix_alpha = 1 - global_avg_alpha
    distance_matrix_beta = 1 - global_avg_beta
    distance_matrix_gamma = 1 - global_avg_gamma
    
    utils.draw_projection(distance_matrix_alpha)
    utils.draw_projection(distance_matrix_beta)
    utils.draw_projection(distance_matrix_gamma)
    
    # Perform hierarchical clustering for each band
    clusters_alpha_dict, clustered_alpha_channels = hierarchical_clustering(distance_matrix_alpha, threshold=0.5, parse=True, verbose=True)
    clusters_beta_dict, clustered_beta_channels = hierarchical_clustering(distance_matrix_beta, threshold=0.5, parse=True, verbose=True)
    clusters_gamma_dict, clustered_gamma_channels = hierarchical_clustering(distance_matrix_gamma, threshold=0.5, parse=True, verbose=True)
    
    # Load channel distribution data
    channel_distribution = utils.get_distribution()
    
    # Plot clusters in 3D space
    plot_3d_channels_(channel_distribution, clusters_alpha_dict)
    plot_3d_channels_(channel_distribution, clusters_beta_dict)
    plot_3d_channels_(channel_distribution, clusters_gamma_dict)
    
    # Extract channel names for alpha clusters
    clustered_alpha_channel_names = [channel_distribution["channel"][idx] for idx in clustered_alpha_channels]
    clustered_beta_channel_names = [channel_distribution["channel"][idx] for idx in clustered_beta_channels]
    clustered_gamma_channel_names = [channel_distribution["channel"][idx] for idx in clustered_gamma_channels]
