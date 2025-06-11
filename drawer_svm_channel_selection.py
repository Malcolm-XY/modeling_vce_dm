# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:19:37 2025

@author: 18307
"""

import numpy as np
import matplotlib.pyplot as plt

# %% executer
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
    plt.ylim(60, ylim)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# %% svm eval; channel feature
def plot_polyline_de():
    # %% argument: Averaged Performance across Decay Models
    title = "Selection Rate vs Performance of Channel Weights: Averaged Performance across Decay Models"
    data_avg = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: Basic AFM x Linear RCN": 
            [82.7303638728132, 81.24505938, 81.21768231, 76.8912575, 77.63241897, 76.8912575, 75.68518264, 73.97455654, 71.20771847],
        "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN": 
            [82.7303638728132, 82.16398642, 79.39098419, 77.00212104, 79.19440809, 77.00212104, 75.99709587, 74.9220841, 74.73507886],
        "Functional Node Strength of RCM: Advanced AFM x Linear RCN": 
            [82.7303638728132, 81.27247847, 81.30748701, 76.5630309, 77.11806125, 76.5630309, 76.03396223, 74.29682742, 72.09827075],
        "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN": 
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
        "Functional Node Strength of RCM: POWERLAW Decay Model x Basic AFM x Linear RCN": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: SIGMOID Decay Model x Basic AFM x Linear-Ratio RCN": 
            [82.73036387, 81.55022679, 80.13331719, 79.99490768, 78.82903831, 76.86251611, 75.71671035, 75.08122619, 73.93995334],
        "Functional Node Strength of RCM: POWERLAW Decay Model x Advanced AFM x Linear RCN": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: RATIONAL-QUADRATIC Decay Model x Advanced AFM x Linear-Ratio RCN": 
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
        "Functional Node Strength of RCM: POWERLAW Decay x Basic AFM x Linear RCN": 
            [82.73036387, 81.55022679, 80.95286868, 80.18097042, 79.45973581, 77.58519249, 75.49122974, 74.20914252, 69.05069536],
        "Functional Node Strength of RCM: POWERLAW Decay x Basic AFM x Linear-Ratio RCN": 
            [82.73036387, 81.55022679, 80.13331719, 80.61317139, 78.82903831, 76.86251611, 74.49475053, 75.08122619, 73.93995334],
        "Functional Node Strength of RCM: POWERLAW Decay x Advanced AFM x Linear RCN": 
            [82.73036387, 81.31674726, 81.2289322, 79.65387821, 77.78469249, 78.52125595, 78.24560189, 75.15483698, 72.09827075],
        "Functional Node Strength of RCM: POWERLAW Decay x Advanced AFM x Linear-Ratio RCN": 
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
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Basic AFM x Linear RCN": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Basic AFM x Linear-Ratio RCN": 
            [82.73036387, 81.2794978, 81.07183165, 80.89036526, 79.38600968, 77.00008939, 72.90931582, 68.35619685, 68.91198021],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Advanced AFM x Linear RCN": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Advanced AFM x Linear-Ratio RCN": 
            [82.73036387, 83.83048873, 81.84078294, 80.11859387, 78.81349031, 77.92799246, 78.33333622, 77.84397789, 70.21950017]
    }
    
    plot_selection_rate_vs_accuracy(title, data_powerlaw,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])

    # %% argument: Performance of GENERALIZED GAUSSIAN decay models
    title = "Selection Rate vs Performance of Channel Weights: GENERALIZED GAUSSIAN Decay Model"
    data_powerlaw = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: GENERALIZED GAUSSIAN Decay x Basic AFM x Linear RCN": 
            [82.73036387, 80.23053833, 80.66060548, 78.66971744, 76.63437688, 74.78972425, 74.95384331, 73.10874662, 72.09827075],
        "Functional Node Strength of RCM: GENERALIZED GAUSSIAN Decay x Basic AFM x Linear-Ratio RCN": 
            [82.73036387, 82.40949028, 78.8473372, 79.63100604, 79.81423138, 77.83501011, 76.35364204, 74.85842726, 75.05312906],
        "Functional Node Strength of RCM: GENERALIZED GAUSSIAN Decay x Advanced AFM x Linear RCN": 
            [82.73036387, 80.23053833, 80.66060548, 78.66971744, 76.63437688, 74.78972425, 73.42196732, 72.89722806, 72.09827075],
        "Functional Node Strength of RCM: GENERALIZED GAUSSIAN Decay x Advanced AFM x Linear-Ratio RCN": 
            [82.73036387, 83.4790324, 82.34618523, 81.45928598, 79.40489681, 79.448392, 79.43936655, 78.75316107, 73.90597381]
    }
    
    plot_selection_rate_vs_accuracy(title, data_powerlaw,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])

def plot_bar_chart_de():
    # %% argument: Averaged Performance across Decay Models
    title = "Selection Rate vs Performance of Channel Weights: Averaged Performance across Decay Models"
    data_avg = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: Basic AFM x Linear RCN": 
            [82.7303638728132, 81.24505938, 81.21768231, 76.8912575, 77.63241897, 76.8912575, 75.68518264, 73.97455654, 71.20771847],
        "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN": 
            [82.7303638728132, 82.16398642, 79.39098419, 77.00212104, 79.19440809, 77.00212104, 75.99709587, 74.9220841, 74.73507886],
        "Functional Node Strength of RCM: Advanced AFM x Linear RCN": 
            [82.7303638728132, 81.27247847, 81.30748701, 76.5630309, 77.11806125, 76.5630309, 76.03396223, 74.29682742, 72.09827075],
        "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN": 
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
        "Functional Node Strength of RCM: POWERLAW Decay Model x Basic AFM x Linear RCN": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: SIGMOID Decay Model x Basic AFM x Linear-Ratio RCN": 
            [82.73036387, 81.55022679, 80.13331719, 79.99490768, 78.82903831, 76.86251611, 75.71671035, 75.08122619, 73.93995334],
        "Functional Node Strength of RCM: POWERLAW Decay Model x Advanced AFM x Linear RCN": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: RATIONAL-QUADRATIC Decay Model x Advanced AFM x Linear-Ratio RCN": 
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
        "Functional Node Strength of RCM: POWERLAW Decay x Basic AFM x Linear RCN": 
            [82.73036387, 81.55022679, 80.95286868, 80.18097042, 79.45973581, 77.58519249, 75.49122974, 74.20914252, 69.05069536],
        "Functional Node Strength of RCM: POWERLAW Decay x Basic AFM x Linear-Ratio RCN": 
            [82.73036387, 81.55022679, 80.13331719, 80.61317139, 78.82903831, 76.86251611, 74.49475053, 75.08122619, 73.93995334],
        "Functional Node Strength of RCM: POWERLAW Decay x Advanced AFM x Linear RCN": 
            [82.73036387, 81.31674726, 81.2289322, 79.65387821, 77.78469249, 78.52125595, 78.24560189, 75.15483698, 72.09827075],
        "Functional Node Strength of RCM: POWERLAW Decay x Advanced AFM x Linear-Ratio RCN": 
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
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Basic AFM x Linear RCN": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Basic AFM x Linear-Ratio RCN": 
            [82.73036387, 81.2794978, 81.07183165, 80.89036526, 79.38600968, 77.00008939, 72.90931582, 68.35619685, 68.91198021],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Advanced AFM x Linear RCN": 
            [82.73036387, 81.70491094, 81.38787821, 79.61666335, 76.63437688, 74.78972425, 76.61047241, 74.31375704, 72.09827075],
        "Functional Node Strength of RCM: EXPONENTIAL Decay x Advanced AFM x Linear-Ratio RCN": 
            [82.73036387, 83.83048873, 81.84078294, 80.11859387, 78.81349031, 77.92799246, 78.33333622, 77.84397789, 70.21950017]
    }
    
    plot_selection_rate_vs_accuracy_bar(title, data_powerlaw,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])

    # %% argument: Performance of GENERALIZED GAUSSIAN decay models
    title = "Selection Rate vs Performance of Channel Weights: GENERALIZED GAUSSIAN Decay Model"
    data_powerlaw = {
        "Functional Node Strength of CM: PCC": 
            [82.7303638728132, 83.73450174, 81.43075055, 75.76393106, 76.69030009, 75.76393106, 76.16170843, 70.955089, 66.34250585],
        "Task-Relevant Channel Importance: MI": 
            [82.7303638728132, 82.29386644, 81.43674253, 80.80022607, 80.43970969, 80.80022607, 81.08382137, 79.61681329, 76.99489327],
        "Functional Node Strength of RCM: GENERALIZED GAUSSIAN Decay x Basic AFM x Linear RCN": 
            [82.73036387, 80.23053833, 80.66060548, 78.66971744, 76.63437688, 74.78972425, 74.95384331, 73.10874662, 72.09827075],
        "Functional Node Strength of RCM: GENERALIZED GAUSSIAN Decay x Basic AFM x Linear-Ratio RCN": 
            [82.73036387, 82.40949028, 78.8473372, 79.63100604, 79.81423138, 77.83501011, 76.35364204, 74.85842726, 75.05312906],
        "Functional Node Strength of RCM: GENERALIZED GAUSSIAN Decay x Advanced AFM x Linear RCN": 
            [82.73036387, 80.23053833, 80.66060548, 78.66971744, 76.63437688, 74.78972425, 73.42196732, 72.89722806, 72.09827075],
        "Functional Node Strength of RCM: GENERALIZED GAUSSIAN Decay x Advanced AFM x Linear-Ratio RCN": 
            [82.73036387, 83.4790324, 82.34618523, 81.45928598, 79.40489681, 79.448392, 79.43936655, 78.75316107, 73.90597381]
    }
    
    plot_selection_rate_vs_accuracy_bar(title, data_powerlaw,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])

def plot_polyline_psd():
    # %% argument: Averaged Performance across Decay Models
    title = "Selection Rate vs Performance of Channel Weights: Averaged Performance across Decay Models"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [68.33785183, 71.85062155, 72.6802481, 69.20574861, 71.99186844, 69.73600118, 70.51713827, 63.25720811, 60.18089545],
    "Task-Relevant Channel Importance: MI":
        [68.33785183, 70.78851316, 78.10723276, 77.80285873, 78.32093127, 78.40085699, 76.15538773, 75.71899411, 72.74207678],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [68.33785183, 66.74986725, 69.512745, 69.20400613, 65.63313491, 64.33025743, 61.47606648, 61.52793956, 60.62592155],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [68.33785183, 67.73103491, 65.12064818, 68.57217911, 65.22959871, 63.64286229, 65.64229959, 67.37023677, 67.46176746],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [68.33785183, 66.05640682, 69.62572588, 69.09431083, 65.90119456, 64.1084974, 62.42845979, 61.24745683, 59.62744776],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [68.33785183, 66.81701234, 67.49647529, 69.36976552, 71.28037028, 72.7412249, 73.26481486, 75.42053626, 72.89939895]
    }
    plot_selection_rate_vs_accuracy(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of decay models with minimum MSE; 
    # MSE between channel weights by models and Task-Relevant Channel Importance
    title = "Selection Rate vs Performance of Channel Weights: Decay Models with Minimum MSE"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [68.33785183, 71.85062155, 72.6802481, 69.20574861, 71.99186844, 69.73600118, 70.51713827, 63.25720811, 60.18089545],
    "Task-Relevant Channel Importance: MI":
        [68.33785183, 70.78851316, 78.10723276, 77.80285873, 78.32093127, 78.40085699, 76.15538773, 75.71899411, 72.74207678],
    "Functional Node Strength of RCM: POWERLAW Decay Model x Basic AFM x Linear RCN": 
        [68.33785183, 63.23632557, 69.34801051, 69.65657719, 65.39498323, 63.97029963, 61.5640043, 59.61496207, 59.62744776],
    "Functional Node Strength of RCM: SIGMOID Decay Model x Basic AFM x Linear-Ratio RCN": 
        [68.33785183, 67.78529803, 70.12690998, 68.70704764, 65.45305755, 62.74803992, 66.04953907, 67.12635922, 66.31478934],
    "Functional Node Strength of RCM: POWERLAW Decay Model x Advanced AFM x Linear RCN": 
        [68.33785183, 63.23632557, 69.34801051, 69.65657719, 65.39498323, 63.97029963, 61.5640043, 59.61496207, 59.62744776],
    "Functional Node Strength of RCM: RATIONAL-QUADRATIC Decay Model x Advanced AFM x Linear-Ratio RCN": 
        [68.33785183, 67.11928304, 71.50902113, 71.94616447, 71.64223451, 71.0866876, 69.95747368, 68.19409626, 66.27696318]
    }
    plot_selection_rate_vs_accuracy(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of POWERLAW decay models
    title = "Selection Rate vs Performance of Channel Weights: POWERLAW Decay Model"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [68.33785183, 71.85062155, 72.6802481, 69.20574861, 71.99186844, 69.73600118, 70.51713827, 63.25720811, 60.18089545],
    "Task-Relevant Channel Importance: MI":
        [68.33785183, 70.78851316, 78.10723276, 77.80285873, 78.32093127, 78.40085699, 76.15538773, 75.71899411, 72.74207678],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [68.33785183, 63.23632557, 69.34801051, 69.65657719, 65.39498323, 63.97029963, 61.5640043, 59.61496207, 59.62744776],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [68.33785183, 67.59537712, 60.85354833, 66.12024325, 64.00371976, 61.32663201, 65.0794557, 67.97993062, 70.32921277],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [68.33785183, 63.23632557, 69.34801051, 69.65657719, 65.39498323, 63.97029963, 61.5640043, 59.61496207, 59.62744776],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [68.33785183, 67.61030805, 70.73560037, 72.29345698, 69.5124756, 71.4824119, 75.65376863, 80.14010502, 78.19135114]
    }
    plot_selection_rate_vs_accuracy(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of EXPONENTIAL decay models
    title = "Selection Rate vs Performance of Channel Weights: EXPONENTIAL Decay Model"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [68.33785183, 71.85062155, 72.6802481, 69.20574861, 71.99186844, 69.73600118, 70.51713827, 63.25720811, 60.18089545],
    "Task-Relevant Channel Importance: MI":
        [68.33785183, 70.78851316, 78.10723276, 77.80285873, 78.32093127, 78.40085699, 76.15538773, 75.71899411, 72.74207678],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [68.33785183, 67.59537712, 70.06158646, 69.71239082, 67.06120872, 63.48641424, 61.29790627, 62.36250602, 62.5409101],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [68.33785183, 67.59537712, 60.85354833, 68.6851155, 64.00371976, 61.32663201, 64.1689461, 67.97993062, 70.32921277],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [68.33785183, 67.23954359, 69.89106019, 70.70890463, 66.46248382, 63.70696402, 66.9317151, 62.60468228, 59.62744776],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [68.33785183, 67.78529803, 61.85923177, 69.19464701, 64.09924538, 62.03085955, 65.0794557, 67.97993062, 66.31478934]
    }
    plot_selection_rate_vs_accuracy(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of GENERALIZED GAUSSIAN decay models
    title = "Selection Rate vs Performance of Channel Weights: GENERALIZED GAUSSIAN Decay Model"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [68.33785183, 71.85062155, 72.6802481, 69.20574861, 71.99186844, 69.73600118, 70.51713827, 63.25720811, 60.18089545],
    "Task-Relevant Channel Importance: MI":
        [68.33785183, 70.78851316, 78.10723276, 77.80285873, 78.32093127, 78.40085699, 76.15538773, 75.71899411, 72.74207678],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [68.33785183, 66.96402218, 69.11618613, 67.62443735, 65.39498323, 63.97029963, 58.83228517, 56.77051993, 59.62744776],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [68.33785183, 67.78529803, 70.12690998, 69.19464701, 67.47410156, 64.04528298, 66.04953907, 67.12635922, 66.31478934],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [68.33785183, 66.96402218, 69.11618613, 67.62443735, 65.39498323, 63.97029963, 60.08731909, 58.2850486, 59.62744776],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [68.33785183, 66.38154309, 67.1434701, 66.52942211, 72.80970712, 76.09973558, 75.54075151, 78.08801691, 75.58578073]
    }
    plot_selection_rate_vs_accuracy(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
def plot_bar_chart_psd():
    # %% argument: Averaged Performance across Decay Models
    title = "Selection Rate vs Performance of Channel Weights: Averaged Performance across Decay Models"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [68.33785183, 71.85062155, 72.6802481, 69.20574861, 71.99186844, 69.73600118, 70.51713827, 63.25720811, 60.18089545],
    "Task-Relevant Channel Importance: MI":
        [68.33785183, 70.78851316, 78.10723276, 77.80285873, 78.32093127, 78.40085699, 76.15538773, 75.71899411, 72.74207678],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [68.33785183, 66.74986725, 69.512745, 69.20400613, 65.63313491, 64.33025743, 61.47606648, 61.52793956, 60.62592155],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [68.33785183, 67.73103491, 65.12064818, 68.57217911, 65.22959871, 63.64286229, 65.64229959, 67.37023677, 67.46176746],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [68.33785183, 66.05640682, 69.62572588, 69.09431083, 65.90119456, 64.1084974, 62.42845979, 61.24745683, 59.62744776],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [68.33785183, 66.81701234, 67.49647529, 69.36976552, 71.28037028, 72.7412249, 73.26481486, 75.42053626, 72.89939895]
    }
    plot_selection_rate_vs_accuracy_bar(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of decay models with minimum MSE; 
    # MSE between channel weights by models and Task-Relevant Channel Importance
    title = "Selection Rate vs Performance of Channel Weights: Decay Models with Minimum MSE"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [68.33785183, 71.85062155, 72.6802481, 69.20574861, 71.99186844, 69.73600118, 70.51713827, 63.25720811, 60.18089545],
    "Task-Relevant Channel Importance: MI":
        [68.33785183, 70.78851316, 78.10723276, 77.80285873, 78.32093127, 78.40085699, 76.15538773, 75.71899411, 72.74207678],
    "Functional Node Strength of RCM: POWERLAW Decay Model x Basic AFM x Linear RCN": 
        [68.33785183, 63.23632557, 69.34801051, 69.65657719, 65.39498323, 63.97029963, 61.5640043, 59.61496207, 59.62744776],
    "Functional Node Strength of RCM: SIGMOID Decay Model x Basic AFM x Linear-Ratio RCN": 
        [68.33785183, 67.78529803, 70.12690998, 68.70704764, 65.45305755, 62.74803992, 66.04953907, 67.12635922, 66.31478934],
    "Functional Node Strength of RCM: POWERLAW Decay Model x Advanced AFM x Linear RCN": 
        [68.33785183, 63.23632557, 69.34801051, 69.65657719, 65.39498323, 63.97029963, 61.5640043, 59.61496207, 59.62744776],
    "Functional Node Strength of RCM: RATIONAL-QUADRATIC Decay Model x Advanced AFM x Linear-Ratio RCN": 
        [68.33785183, 67.11928304, 71.50902113, 71.94616447, 71.64223451, 71.0866876, 69.95747368, 68.19409626, 66.27696318]
    }
    plot_selection_rate_vs_accuracy_bar(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of POWERLAW decay models
    title = "Selection Rate vs Performance of Channel Weights: POWERLAW Decay Model"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [68.33785183, 71.85062155, 72.6802481, 69.20574861, 71.99186844, 69.73600118, 70.51713827, 63.25720811, 60.18089545],
    "Task-Relevant Channel Importance: MI":
        [68.33785183, 70.78851316, 78.10723276, 77.80285873, 78.32093127, 78.40085699, 76.15538773, 75.71899411, 72.74207678],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [68.33785183, 63.23632557, 69.34801051, 69.65657719, 65.39498323, 63.97029963, 61.5640043, 59.61496207, 59.62744776],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [68.33785183, 67.59537712, 60.85354833, 66.12024325, 64.00371976, 61.32663201, 65.0794557, 67.97993062, 70.32921277],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [68.33785183, 63.23632557, 69.34801051, 69.65657719, 65.39498323, 63.97029963, 61.5640043, 59.61496207, 59.62744776],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [68.33785183, 67.61030805, 70.73560037, 72.29345698, 69.5124756, 71.4824119, 75.65376863, 80.14010502, 78.19135114]
    }
    plot_selection_rate_vs_accuracy_bar(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of EXPONENTIAL decay models
    title = "Selection Rate vs Performance of Channel Weights: EXPONENTIAL Decay Model"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [68.33785183, 71.85062155, 72.6802481, 69.20574861, 71.99186844, 69.73600118, 70.51713827, 63.25720811, 60.18089545],
    "Task-Relevant Channel Importance: MI":
        [68.33785183, 70.78851316, 78.10723276, 77.80285873, 78.32093127, 78.40085699, 76.15538773, 75.71899411, 72.74207678],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [68.33785183, 67.59537712, 70.06158646, 69.71239082, 67.06120872, 63.48641424, 61.29790627, 62.36250602, 62.5409101],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [68.33785183, 67.59537712, 60.85354833, 68.6851155, 64.00371976, 61.32663201, 64.1689461, 67.97993062, 70.32921277],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [68.33785183, 67.23954359, 69.89106019, 70.70890463, 66.46248382, 63.70696402, 66.9317151, 62.60468228, 59.62744776],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [68.33785183, 67.78529803, 61.85923177, 69.19464701, 64.09924538, 62.03085955, 65.0794557, 67.97993062, 66.31478934]
    }
    plot_selection_rate_vs_accuracy_bar(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of GENERALIZED GAUSSIAN decay models
    title = "Selection Rate vs Performance of Channel Weights: GENERALIZED GAUSSIAN Decay Model"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [68.33785183, 71.85062155, 72.6802481, 69.20574861, 71.99186844, 69.73600118, 70.51713827, 63.25720811, 60.18089545],
    "Task-Relevant Channel Importance: MI":
        [68.33785183, 70.78851316, 78.10723276, 77.80285873, 78.32093127, 78.40085699, 76.15538773, 75.71899411, 72.74207678],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [68.33785183, 66.96402218, 69.11618613, 67.62443735, 65.39498323, 63.97029963, 58.83228517, 56.77051993, 59.62744776],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [68.33785183, 67.78529803, 70.12690998, 69.19464701, 67.47410156, 64.04528298, 66.04953907, 67.12635922, 66.31478934],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [68.33785183, 66.96402218, 69.11618613, 67.62443735, 65.39498323, 63.97029963, 60.08731909, 58.2850486, 59.62744776],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [68.33785183, 66.38154309, 67.1434701, 66.52942211, 72.80970712, 76.09973558, 75.54075151, 78.08801691, 75.58578073]
    }    
    plot_selection_rate_vs_accuracy_bar(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
# %% subnetworks cnn eval; functional connectivity
def plot_polyline_pcc():
    # %% argument: Averaged Performance across Decay Models
    title = "Selection Rate vs Performance of Channel Weights: Averaged Performance across Decay Models"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [92.53097345, 92.95575221, 92.23266637, 90.44926571, 86.42140503, 79.28418643, 78.06388175, 66.2839817, 56.12908993],
    "Task-Relevant Channel Importance: MI":
        [92.53097345, 92.18485742, 91.48088363, 91.00104095, 90.13789047, 88.75227871, 87.75053994, 85.38950452, 76.22752215],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [92.5402543, 92.35152306, 91.41748384, 90.56903689, 87.63336229, 85.30051628, 81.54298254, 74.78526878, 62.93890168],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [92.5402543, 92.50637611, 90.93600497, 89.59301757, 85.57178325, 81.28861306, 77.3641072, 72.64724485, 62.64748377],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [92.5402543, 92.2501175, 91.14183225, 90.29891592, 87.53566249, 85.45713239, 82.04800421, 74.39458572, 61.40150825],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [92.5402543, 92.07693673, 91.65154914, 91.29863539, 89.14349152, 87.78587226, 85.72451648, 79.63020029, 66.38774598]
    } 
    plot_selection_rate_vs_accuracy(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of decay models with minimum MSE; 
    # MSE between channel weights by models and Task-Relevant Channel Importance
    title = "Selection Rate vs Performance of Channel Weights: Decay Models with Minimum MSE"
    data_min_mse = {
    "Functional Node Strength of CM: PCC":
        [92.53097345, 92.95575221, 92.23266637, 90.44926571, 86.42140503, 79.28418643, 78.06388175, 66.2839817, 56.12908993],
    "Task-Relevant Channel Importance: MI":
        [92.53097345, 92.18485742, 91.48088363, 91.00104095, 90.13789047, 88.75227871, 87.75053994, 85.38950452, 76.22752215],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [92.5402543, 92.51923171, 90.79870356, 90.38044735, 86.80530685, 82.21402723, 81.16984864, 72.84413072, 57.5067489],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [92.5402543, 92.58999357, 90.80660444, 88.69301638, 85.2342725, 80.50317621, 77.40367996, 71.84381064, 61.68469162],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [92.5402543, 91.95873667, 91.06010721, 90.63344262, 85.985888, 83.74006696, 80.52885694, 70.54336687, 57.4629769],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [92.5402543, 89.60015802, 88.10312085, 87.5445693, 85.66425892, 86.0674631, 84.35433409, 72.64253151, 56.15265415]
    }
    plot_selection_rate_vs_accuracy(title, data_min_mse,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of POWERLAW decay models
    title = "Selection Rate vs Performance of Channel Weights: POWERLAW Decay Model"
    data_powerlaw = {
    "Functional Node Strength of CM: PCC":
        [92.53097345, 92.95575221, 92.23266637, 90.44926571, 86.42140503, 79.28418643, 78.06388175, 66.2839817, 56.12908993],
    "Task-Relevant Channel Importance: MI":
        [92.53097345, 92.18485742, 91.48088363, 91.00104095, 90.13789047, 88.75227871, 87.75053994, 85.38950452, 76.22752215],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [92.5402543, 92.51923171, 90.79870356, 90.38044735, 86.80530685, 82.21402723, 81.16984864, 72.84413072, 57.5067489],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [92.5402543, 92.30482963, 90.96906836, 88.90684175, 85.86946831, 81.23695995, 78.08331675, 73.73483046, 64.59898442],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [92.5402543, 91.95873667, 91.06010721, 90.63344262, 85.985888, 83.74006696, 80.52885694, 70.54336687, 57.4629769],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [92.5402543, 91.96268134, 91.33341407, 90.95187675, 89.83427192, 88.69703025, 88.51773228, 85.83904705, 71.61617315]
    }
    plot_selection_rate_vs_accuracy(title, data_powerlaw,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of EXPONENTIAL decay models
    title = "Selection Rate vs Performance of Channel Weights: EXPONENTIAL Decay Model"
    data_exponential = {
    "Functional Node Strength of CM: PCC":
        [92.53097345, 92.95575221, 92.23266637, 90.44926571, 86.42140503, 79.28418643, 78.06388175, 66.2839817, 56.12908993],
    "Task-Relevant Channel Importance: MI":
        [92.53097345, 92.18485742, 91.48088363, 91.00104095, 90.13789047, 88.75227871, 87.75053994, 85.38950452, 76.22752215],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [92.5402543, 92.28908555, 91.40070416, 90.78494912, 88.20234316, 84.71827611, 81.06229293, 74.2491025, 62.70019637],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [92.5402543, 92.48252436, 91.24886317, 88.86195094, 85.53752195, 79.9489788, 76.53315917, 74.18452293, 65.19917416],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [92.5402543, 91.91348253, 91.28028212, 90.02571533, 87.67330744, 86.34135243, 81.35238771, 70.36928808, 58.05692091],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [92.5402543, 92.5054196, 90.95402786, 90.13812115, 83.55062472, 78.95119047, 77.98806795, 73.37150552, 61.94430748]
    }
    plot_selection_rate_vs_accuracy(title, data_exponential,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])
    
    # %% argument: Performance of GENERALIZED GAUSSIAN decay models
    title = "Selection Rate vs Performance of Channel Weights: GENERALIZED GAUSSIAN Decay Model"
    data_gg = {
    "Functional Node Strength of CM: PCC":
        [92.53097345, 92.95575221, 92.23266637, 90.44926571, 86.42140503, 79.28418643, 78.06388175, 66.2839817, 56.12908993],
    "Task-Relevant Channel Importance: MI":
        [92.53097345, 92.18485742, 91.48088363, 91.00104095, 90.13789047, 88.75227871, 87.75053994, 85.38950452, 76.22752215],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [92.5402543, 92.0669902, 91.1269244, 89.97266412, 86.905994, 83.92330383, 82.94038299, 76.38216017, 64.20707215],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [92.5402543, 92.81420543, 90.94222845, 90.13469551, 87.04513015, 81.14596724, 77.1187467, 72.56822868, 61.28117602],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [92.5402543, 92.12210025, 91.05023977, 89.64642139, 86.89005384, 83.40004095, 78.98070629, 72.44580547, 57.1420514],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [92.5402543, 93.01870547, 92.61356932, 92.82202557, 91.54614948, 90.73923938, 87.36320095, 81.29539183, 69.36209944]
    }    
    plot_selection_rate_vs_accuracy(title, data_gg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'])

def plot_bar_chart_pcc():
    # %% argument: Averaged Performance across Decay Models
    title = "Selection Rate vs Performance of Channel Weights: Averaged Performance across Decay Models"
    data_avg = {
    "Functional Node Strength of CM: PCC":
        [92.53097345, 92.95575221, 92.23266637, 90.44926571, 86.42140503, 79.28418643, 78.06388175, 66.2839817, 56.12908993],
    "Task-Relevant Channel Importance: MI":
        [92.53097345, 92.18485742, 91.48088363, 91.00104095, 90.13789047, 88.75227871, 87.75053994, 85.38950452, 76.22752215],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [92.5402543, 92.35152306, 91.41748384, 90.56903689, 87.63336229, 85.30051628, 81.54298254, 74.78526878, 62.93890168],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [92.5402543, 92.50637611, 90.93600497, 89.59301757, 85.57178325, 81.28861306, 77.3641072, 72.64724485, 62.64748377],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [92.5402543, 92.2501175, 91.14183225, 90.29891592, 87.53566249, 85.45713239, 82.04800421, 74.39458572, 61.40150825],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [92.5402543, 92.07693673, 91.65154914, 91.29863539, 89.14349152, 87.78587226, 85.72451648, 79.63020029, 66.38774598]
    } 
    plot_selection_rate_vs_accuracy_bar(title, data_avg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'], ylim=100)
    
    # %% argument: Performance of decay models with minimum MSE; 
    # MSE between channel weights by models and Task-Relevant Channel Importance
    title = "Selection Rate vs Performance of Channel Weights: Decay Models with Minimum MSE"
    data_min_mse = {
    "Functional Node Strength of CM: PCC":
        [92.53097345, 92.95575221, 92.23266637, 90.44926571, 86.42140503, 79.28418643, 78.06388175, 66.2839817, 56.12908993],
    "Task-Relevant Channel Importance: MI":
        [92.53097345, 92.18485742, 91.48088363, 91.00104095, 90.13789047, 88.75227871, 87.75053994, 85.38950452, 76.22752215],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [92.5402543, 92.51923171, 90.79870356, 90.38044735, 86.80530685, 82.21402723, 81.16984864, 72.84413072, 57.5067489],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [92.5402543, 92.58999357, 90.80660444, 88.69301638, 85.2342725, 80.50317621, 77.40367996, 71.84381064, 61.68469162],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [92.5402543, 91.95873667, 91.06010721, 90.63344262, 85.985888, 83.74006696, 80.52885694, 70.54336687, 57.4629769],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [92.5402543, 89.60015802, 88.10312085, 87.5445693, 85.66425892, 86.0674631, 84.35433409, 72.64253151, 56.15265415]
    }
    plot_selection_rate_vs_accuracy_bar(title, data_min_mse,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'], ylim=100)
    
    # %% argument: Performance of POWERLAW decay models
    title = "Selection Rate vs Performance of Channel Weights: POWERLAW Decay Model"
    data_powerlaw = {
    "Functional Node Strength of CM: PCC":
        [92.53097345, 92.95575221, 92.23266637, 90.44926571, 86.42140503, 79.28418643, 78.06388175, 66.2839817, 56.12908993],
    "Task-Relevant Channel Importance: MI":
        [92.53097345, 92.18485742, 91.48088363, 91.00104095, 90.13789047, 88.75227871, 87.75053994, 85.38950452, 76.22752215],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [92.5402543, 92.51923171, 90.79870356, 90.38044735, 86.80530685, 82.21402723, 81.16984864, 72.84413072, 57.5067489],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [92.5402543, 92.30482963, 90.96906836, 88.90684175, 85.86946831, 81.23695995, 78.08331675, 73.73483046, 64.59898442],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [92.5402543, 91.95873667, 91.06010721, 90.63344262, 85.985888, 83.74006696, 80.52885694, 70.54336687, 57.4629769],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [92.5402543, 91.96268134, 91.33341407, 90.95187675, 89.83427192, 88.69703025, 88.51773228, 85.83904705, 71.61617315]
    }
    plot_selection_rate_vs_accuracy_bar(title, data_powerlaw,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'], ylim=100)
    
    # %% argument: Performance of EXPONENTIAL decay models
    title = "Selection Rate vs Performance of Channel Weights: EXPONENTIAL Decay Model"
    data_exponential = {
    "Functional Node Strength of CM: PCC":
        [92.53097345, 92.95575221, 92.23266637, 90.44926571, 86.42140503, 79.28418643, 78.06388175, 66.2839817, 56.12908993],
    "Task-Relevant Channel Importance: MI":
        [92.53097345, 92.18485742, 91.48088363, 91.00104095, 90.13789047, 88.75227871, 87.75053994, 85.38950452, 76.22752215],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [92.5402543, 92.28908555, 91.40070416, 90.78494912, 88.20234316, 84.71827611, 81.06229293, 74.2491025, 62.70019637],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [92.5402543, 92.48252436, 91.24886317, 88.86195094, 85.53752195, 79.9489788, 76.53315917, 74.18452293, 65.19917416],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [92.5402543, 91.91348253, 91.28028212, 90.02571533, 87.67330744, 86.34135243, 81.35238771, 70.36928808, 58.05692091],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [92.5402543, 92.5054196, 90.95402786, 90.13812115, 83.55062472, 78.95119047, 77.98806795, 73.37150552, 61.94430748]
    }
    plot_selection_rate_vs_accuracy_bar(title, data_exponential,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'], ylim=100)
    
    # %% argument: Performance of GENERALIZED GAUSSIAN decay models
    title = "Selection Rate vs Performance of Channel Weights: GENERALIZED GAUSSIAN Decay Model"
    data_gg = {
    "Functional Node Strength of CM: PCC":
        [92.53097345, 92.95575221, 92.23266637, 90.44926571, 86.42140503, 79.28418643, 78.06388175, 66.2839817, 56.12908993],
    "Task-Relevant Channel Importance: MI":
        [92.53097345, 92.18485742, 91.48088363, 91.00104095, 90.13789047, 88.75227871, 87.75053994, 85.38950452, 76.22752215],
    "Functional Node Strength of RCM: Basic AFM x Linear RCN":
        [92.5402543, 92.0669902, 91.1269244, 89.97266412, 86.905994, 83.92330383, 82.94038299, 76.38216017, 64.20707215],
    "Functional Node Strength of RCM: Basic AFM x Linear-Ratio RCN":
        [92.5402543, 92.81420543, 90.94222845, 90.13469551, 87.04513015, 81.14596724, 77.1187467, 72.56822868, 61.28117602],
    "Functional Node Strength of RCM: Advanced AFM x Linear RCN":
        [92.5402543, 92.12210025, 91.05023977, 89.64642139, 86.89005384, 83.40004095, 78.98070629, 72.44580547, 57.1420514],
    "Functional Node Strength of RCM: Advanced AFM x Linear-Ratio RCN":
        [92.5402543, 93.01870547, 92.61356932, 92.82202557, 91.54614948, 90.73923938, 87.36320095, 81.29539183, 69.36209944]
    }    
    plot_selection_rate_vs_accuracy_bar(title, data_gg,
        colors=['slategrey', 'steelblue', 'indianred', 'red', 'sandybrown', 'darkorange'], ylim=100)

if __name__ == '__main__':
    plot_polyline_de()
    plot_bar_chart_de()
    
    plot_polyline_psd()
    plot_bar_chart_psd()
    
    plot_polyline_pcc()
    plot_bar_chart_pcc()