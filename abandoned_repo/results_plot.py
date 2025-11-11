# -*- coding: utf-8 -*-
"""
Created on Thu May 29 10:27:29 2025

@author: 18307
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% ploter
def plot_selection_rate_vs_accuracy(
    title: str,
    data: dict,
    selection_rate: list = [1, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1],
    colors: list = None
):
    """
    绘制 selection_rate vs accuracy 折线图

    Args:
        title (str): 图表标题
        data (dict): 各方法名称与对应 accuracy 列表
        selection_rate (list): selection rate 列表，默认从 1 到 0.1
        colors (list): 可选颜色列表，若为 None 则自动生成颜色
    """
    if colors is None:
        # 使用 matplotlib 的 tab10 色板自动配色
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(data))]

    plt.figure(figsize=(10, 6))
    for idx, (label, accuracies) in enumerate(data.items()):
        plt.plot(selection_rate, accuracies, marker='o', label=label,
                 color=colors[idx % len(colors)], linewidth=2)

    for x in selection_rate:
        plt.axvline(x=x, linestyle=':', color='gray', alpha=0.5)

    plt.gca().invert_xaxis()
    plt.title(title, fontsize=14)
    plt.xlabel("Selection Rate", fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.xticks(selection_rate)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_selection_rate_vs_accuracy_bar(
    title: str,
    data: dict,
    selection_rate: list = [1, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1],
    colors: list = None,
    ymin=60,
    ylim=90
):
    """
    绘制 selection_rate vs accuracy 的柱状图

    Args:
        title (str): 图表标题
        data (dict): 各方法名称与对应 accuracy 列表
        selection_rate (list): selection rate 列表
        colors (list): 可选颜色列表，若为 None 则自动生成颜色
    """
    method_names = list(data.keys())
    n_methods = len(method_names)
    n_rates = len(selection_rate)

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(n_methods)]

    bar_width = 0.8 / n_methods  # 自动适应每组中柱子的宽度
    x = np.arange(n_rates)  # 每组位置的基准点

    plt.figure(figsize=(12, 6))

    for i, method in enumerate(method_names):
        accuracies = data[method]
        offset = (i - n_methods / 2) * bar_width + bar_width / 2
        plt.bar(
            x + offset,
            accuracies,
            width=bar_width,
            label=method,
            color=colors[i],
            edgecolor='black'
        )

    plt.xticks(ticks=x, labels=[str(r) for r in selection_rate])
    plt.xlabel("Selection Rate", fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.ylim(ymin, ylim)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# %% Excel loader
# Load the Excel file
def load_summary(evaluation='cnn_subnetwork', feature='pcc'):
    evaluation, feature = evaluation.lower(), feature.lower()
    
    if evaluation == 'cnn_subnetwork':
        folder_name = 'results_cnn_subnetwork_evaluation'
    elif evaluation == 'svm_channel_selection':
        folder_name = 'results_svm_channel_selection_evaluation'
    
    match feature:
        case 'pcc':
            folder_secondary_name = f'{feature}_subnetwork_10_15_evaluation'
        case 'plv':
            folder_secondary_name = f'{feature}_subnetwork_10_15_evaluation'
        case 'de_lds':
            folder_secondary_name = 'de_LDS_10_15_evaluation'
        case 'psd_lds':
            folder_secondary_name = 'psd_LDS_10_15_evaluation'
    
    path_current = os.getcwd()
    file_path = os.path.join(path_current, folder_name, folder_secondary_name, 'summary.xlsx')
    
    # Prepare a summary DataFrame
    summary = {}
    sheet_names = ['argument_average', 'argument_exponential', 'argument_powerlaw', 'argument_generalized_gaussian', 'argument_minimumMSE']
    for sheet_name in sheet_names:
        summary[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    
    data_average = {
    "Functional Node Strength of CM":
        summary['argument_average'][2:][1],
    "Task-Relevant Channel Importance: MI":
        summary['argument_average'][2:][2],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        summary['argument_average'][2:][3],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        summary['argument_average'][2:][4],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        summary['argument_average'][2:][5],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        summary['argument_average'][2:][6],
    }
    
    data_minimumMSE = {
    "Functional Node Strength of CM":
        summary['argument_minimumMSE'][3:][1],
    "Task-Relevant Channel Importance: MI":
        summary['argument_minimumMSE'][3:][2],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        summary['argument_minimumMSE'][3:][3],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        summary['argument_minimumMSE'][3:][4],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        summary['argument_minimumMSE'][3:][5],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        summary['argument_minimumMSE'][3:][6],
    }
        
    data_powerlaw = {
    "Functional Node Strength of CM":
        summary['argument_powerlaw'][2:][1],
    "Task-Relevant Channel Importance: MI":
        summary['argument_powerlaw'][2:][2],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        summary['argument_powerlaw'][2:][3],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        summary['argument_powerlaw'][2:][4],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        summary['argument_powerlaw'][2:][5],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        summary['argument_powerlaw'][2:][6],
    }
        
    data_exponential = {
    "Functional Node Strength of CM":
        summary['argument_exponential'][2:][1],
    "Task-Relevant Channel Importance: MI":
        summary['argument_exponential'][2:][2],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        summary['argument_exponential'][2:][3],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        summary['argument_exponential'][2:][4],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        summary['argument_exponential'][2:][5],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        summary['argument_exponential'][2:][6],
    }
    
    data_generalized_gaussian = {
    "Functional Node Strength of CM":
        summary['argument_generalized_gaussian'][2:][1],
    "Task-Relevant Channel Importance: MI":
        summary['argument_generalized_gaussian'][2:][2],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        summary['argument_generalized_gaussian'][2:][3],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        summary['argument_generalized_gaussian'][2:][4],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        summary['argument_generalized_gaussian'][2:][5],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        summary['argument_generalized_gaussian'][2:][6],
    }
        
    return summary, data_average, data_minimumMSE, data_exponential, data_generalized_gaussian, data_powerlaw
    
# %% drawer
def polyline_drawer(evaluation='cnn_subnetwork', feature='pcc'):
    _, data_average, data_minimumMSE, data_exponential, data_generalized_gaussian, data_powerlaw = load_summary(evaluation, feature)
    
    colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange']
    
    tuples = {
        'average': ("Selection Rate vs Performance of Channel Weights: Averaged Performance across Decay Models", data_average),
        'minimumMSE': ("Selection Rate vs Performance of Channel Weights: Decay Models with Minimum MSE", data_minimumMSE),
        'exponential': ("Selection Rate vs Performance of Channel Weights: EXPONENTIAL Decay Model", data_exponential),
        'generalized_gaussian': ("Selection Rate vs Performance of Channel Weights: GENERALIZED GAUSSIAN Decay Model", data_generalized_gaussian),
        'powerlaw': ("Selection Rate vs Performance of Channel Weights: POWERLAW Decay Model", data_powerlaw)
        }
    
    plot_selection_rate_vs_accuracy(tuples['average'][0], tuples['average'][1], colors=colors)
    plot_selection_rate_vs_accuracy(tuples['minimumMSE'][0], tuples['minimumMSE'][1], colors=colors)
    plot_selection_rate_vs_accuracy(tuples['exponential'][0], tuples['exponential'][1], colors=colors)
    plot_selection_rate_vs_accuracy(tuples['generalized_gaussian'][0], tuples['generalized_gaussian'][1], colors=colors)
    plot_selection_rate_vs_accuracy(tuples['powerlaw'][0], tuples['powerlaw'][1], colors=colors)

def barchart_drawer(evaluation='cnn_subnetwork', feature='pcc', ymin=60, ylim=90):
    _, data_average, data_minimumMSE, data_exponential, data_generalized_gaussian, data_powerlaw = load_summary(evaluation, feature)
    
    colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange']
    
    tuples = {
        'average': ("Selection Rate vs Performance of Channel Weights: Averaged Performance across Decay Models", data_average),
        'minimumMSE': ("Selection Rate vs Performance of Channel Weights: Decay Models with Minimum MSE", data_minimumMSE),
        'exponential': ("Selection Rate vs Performance of Channel Weights: EXPONENTIAL Decay Model", data_exponential),
        'generalized_gaussian': ("Selection Rate vs Performance of Channel Weights: GENERALIZED GAUSSIAN Decay Model", data_generalized_gaussian),
        'powerlaw': ("Selection Rate vs Performance of Channel Weights: POWERLAW Decay Model", data_powerlaw)
        }
    
    plot_selection_rate_vs_accuracy_bar(tuples['average'][0], tuples['average'][1], colors=colors, ymin=ymin, ylim=ylim)
    plot_selection_rate_vs_accuracy_bar(tuples['minimumMSE'][0], tuples['minimumMSE'][1], colors=colors, ymin=ymin, ylim=ylim)
    plot_selection_rate_vs_accuracy_bar(tuples['exponential'][0], tuples['exponential'][1], colors=colors, ymin=ymin, ylim=ylim)
    plot_selection_rate_vs_accuracy_bar(tuples['generalized_gaussian'][0], tuples['generalized_gaussian'][1], colors=colors, ymin=ymin, ylim=ylim)
    plot_selection_rate_vs_accuracy_bar(tuples['powerlaw'][0], tuples['powerlaw'][1], colors=colors, ymin=ymin, ylim=ylim)

# %% usage
if __name__ == '__main__':
    polyline_drawer(evaluation='cnn_subnetwork', feature='pcc')
    barchart_drawer(evaluation='cnn_subnetwork', feature='pcc', ymin=50, ylim=100)
    polyline_drawer(evaluation='cnn_subnetwork', feature='plv')
    barchart_drawer(evaluation='cnn_subnetwork', feature='plv', ymin=50, ylim=100)
    
    polyline_drawer(evaluation='svm_channel_selection', feature='de_LDS')
    barchart_drawer(evaluation='svm_channel_selection', feature='de_LDS', ymin=50, ylim=100)
    polyline_drawer(evaluation='svm_channel_selection', feature='psd_LDS')
    barchart_drawer(evaluation='svm_channel_selection', feature='psd_LDS', ymin=50, ylim=100)