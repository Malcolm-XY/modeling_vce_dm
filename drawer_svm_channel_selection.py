# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:19:37 2025

@author: 18307
"""

import numpy as np
import matplotlib.pyplot as plt

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
    colors: list = None
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
    plt.ylim(60, 90)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_polyline():
    # %% argument: Averaged Performance across Decay Models
    title = "Selection Rate vs Performance of Channel Weights: Averaged Performance across Decay Models"
    data_avg = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: Basic FM x Linear RCM": 
            [82.7303638728132, 81.24505938, 81.21768231, 76.8912575, 77.63241897, 76.8912575, 75.68518264, 73.97455654, 71.20771847],
        "Functional Node Strength of RCM: Basic FM x Linear-Ratio RCM": 
            [82.7303638728132, 82.16398642, 79.39098419, 77.00212104, 79.19440809, 77.00212104, 75.99709587, 74.9220841, 74.73507886],
        "Functional Node Strength of RCM: Advanced FM x Linear RCM": 
            [82.7303638728132, 81.27247847, 81.30748701, 76.5630309, 77.11806125, 76.5630309, 76.03396223, 74.29682742, 72.09827075],
        "Functional Node Strength of RCM: Advanced FM x Linear-Ratio RCM": 
            [82.7303638728132, 83.23265855, 81.30909107, 78.69245862, 79.00868643, 78.69245862, 78.62541569, 78.14617032, 73.8768279]
    }
    
    plot_selection_rate_vs_accuracy(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of decay models with minimum MSE; 
    # MSE between channel weights by models and Task-Relevant Channel Importance
    title = "Selection Rate vs Performance of Channel Weights: Decay Models with Minimum MSE"
    data_min_mse = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: POWERLAW Decay Model x Basic FM x Linear RCM": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: SIGMOID Decay Model x Basic FM x Linear-Ratio RCM": 
            [82.73036387, 81.55022679, 80.13331719, 79.99490768, 78.82903831, 76.86251611, 75.71671035, 75.08122619, 73.93995334],
        "Functional Node Strength of RCM: POWERLAW Decay Model x Advanced FM x Linear RCM": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: RATIONAL-QUADRATIC Decay Model x Advanced FM x Linear-Ratio RCM": 
            [82.73036387, 82.46044804, 79.29159133, 78.64676453, 77.20091004, 78.57108914, 78.57039709, 79.25719081, 77.0307817]
    }
    
    plot_selection_rate_vs_accuracy(title, data_min_mse,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of POWERLAW decay models
    title = "Selection Rate vs Performance of Channel Weights: POWERLAW Decay Model"
    data_powerlaw = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: POWERLAW Decay x Basic FM x Linear RCM": 
            [82.73036387, 81.55022679, 80.95286868, 80.18097042, 79.45973581, 77.58519249, 75.49122974, 74.20914252, 69.05069536],
        "Functional Node Strength of RCM: POWERLAW Decay x Basic FM x Linear-Ratio RCM": 
            [82.73036387, 81.55022679, 80.13331719, 80.61317139, 78.82903831, 76.86251611, 74.49475053, 75.08122619, 73.93995334],
        "Functional Node Strength of RCM: POWERLAW Decay x Advanced FM x Linear RCM": 
            [82.73036387, 81.31674726, 81.2289322, 79.65387821, 77.78469249, 78.52125595, 78.24560189, 75.15483698, 72.09827075],
        "Functional Node Strength of RCM: POWERLAW Decay x Advanced FM x Linear-Ratio RCM": 
            [82.73036387, 82.40949028, 79.15551749, 79.63100604, 78.84415378, 76.38271381, 75.71671035, 75.08122619, 75.05312906]
    }
    
    plot_selection_rate_vs_accuracy(title, data_powerlaw,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of EXPONENTIAL decay models
    title = "Selection Rate vs Performance of Channel Weights: EXPONENTIAL Decay Model"
    data_powerlaw = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Basic FM x Linear RCM": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Basic FM x Linear-Ratio RCM": 
            [82.73036387, 81.2794978, 81.07183165, 80.89036526, 79.38600968, 77.00008939, 72.90931582, 68.35619685, 68.91198021],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Advanced FM x Linear RCM": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Advanced FM x Linear-Ratio RCM": 
            [82.73036387, 83.83048873, 81.84078294, 80.11859387, 78.81349031, 77.92799246, 78.33333622, 77.84397789, 70.21950017]
    }
    
    plot_selection_rate_vs_accuracy(title, data_powerlaw,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])

def plot_bar_chart():
    # %% argument: Averaged Performance across Decay Models
    title = "Selection Rate vs Performance of Channel Weights: Averaged Performance across Decay Models"
    data_avg = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: Basic FM x Linear RCM": 
            [82.7303638728132, 81.24505938, 81.21768231, 76.8912575, 77.63241897, 76.8912575, 75.68518264, 73.97455654, 71.20771847],
        "Functional Node Strength of RCM: Basic FM x Linear-Ratio RCM": 
            [82.7303638728132, 82.16398642, 79.39098419, 77.00212104, 79.19440809, 77.00212104, 75.99709587, 74.9220841, 74.73507886],
        "Functional Node Strength of RCM: Advanced FM x Linear RCM": 
            [82.7303638728132, 81.27247847, 81.30748701, 76.5630309, 77.11806125, 76.5630309, 76.03396223, 74.29682742, 72.09827075],
        "Functional Node Strength of RCM: Advanced FM x Linear-Ratio RCM": 
            [82.7303638728132, 83.23265855, 81.30909107, 78.69245862, 79.00868643, 78.69245862, 78.62541569, 78.14617032, 73.8768279]
    }
    
    plot_selection_rate_vs_accuracy_bar(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of decay models with minimum MSE; 
    # MSE between channel weights by models and Task-Relevant Channel Importance
    title = "Selection Rate vs Performance of Channel Weights: Decay Models with Minimum MSE"
    data_min_mse = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: POWERLAW Decay Model x Basic FM x Linear RCM": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: SIGMOID Decay Model x Basic FM x Linear-Ratio RCM": 
            [82.73036387, 81.55022679, 80.13331719, 79.99490768, 78.82903831, 76.86251611, 75.71671035, 75.08122619, 73.93995334],
        "Functional Node Strength of RCM: POWERLAW Decay Model x Advanced FM x Linear RCM": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: RATIONAL-QUADRATIC Decay Model x Advanced FM x Linear-Ratio RCM": 
            [82.73036387, 82.46044804, 79.29159133, 78.64676453, 77.20091004, 78.57108914, 78.57039709, 79.25719081, 77.0307817]
    }
    
    plot_selection_rate_vs_accuracy_bar(title, data_min_mse,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of POWERLAW decay models
    title = "Selection Rate vs Performance of Channel Weights: POWERLAW Decay Model"
    data_powerlaw = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: POWERLAW Decay x Basic FM x Linear RCM": 
            [82.73036387, 81.55022679, 80.95286868, 80.18097042, 79.45973581, 77.58519249, 75.49122974, 74.20914252, 69.05069536],
        "Functional Node Strength of RCM: POWERLAW Decay x Basic FM x Linear-Ratio RCM": 
            [82.73036387, 81.55022679, 80.13331719, 80.61317139, 78.82903831, 76.86251611, 74.49475053, 75.08122619, 73.93995334],
        "Functional Node Strength of RCM: POWERLAW Decay x Advanced FM x Linear RCM": 
            [82.73036387, 81.31674726, 81.2289322, 79.65387821, 77.78469249, 78.52125595, 78.24560189, 75.15483698, 72.09827075],
        "Functional Node Strength of RCM: POWERLAW Decay x Advanced FM x Linear-Ratio RCM": 
            [82.73036387, 82.40949028, 79.15551749, 79.63100604, 78.84415378, 76.38271381, 75.71671035, 75.08122619, 75.05312906]
    }
    
    plot_selection_rate_vs_accuracy_bar(title, data_powerlaw,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of EXPONENTIAL decay models
    title = "Selection Rate vs Performance of Channel Weights: EXPONENTIAL Decay Model"
    data_powerlaw = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Basic FM x Linear RCM": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Basic FM x Linear-Ratio RCM": 
            [82.73036387, 81.2794978, 81.07183165, 80.89036526, 79.38600968, 77.00008939, 72.90931582, 68.35619685, 68.91198021],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Advanced FM x Linear RCM": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Advanced FM x Linear-Ratio RCM": 
            [82.73036387, 83.83048873, 81.84078294, 80.11859387, 78.81349031, 77.92799246, 78.33333622, 77.84397789, 70.21950017]
    }
    
    plot_selection_rate_vs_accuracy_bar(title, data_powerlaw,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])

if __name__ == '__main__':
    plot_polyline()
    plot_bar_chart()