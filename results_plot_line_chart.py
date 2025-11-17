# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 22:45:31 2025

@author: 18307
"""

# %% -----------------------------
def selection_robust_auc(srs, accuracies):
    aucs = []
    n = len(srs) - 1
    for i in range(n):
        auc = (srs[i]-srs[i+1]) * (accuracies[i]+accuracies[i+1])/2
        aucs.append(auc)
        
    auc = np.sum(aucs) * 1/(srs[0]-srs[-1])
    
    return auc

def balanced_performance_efficiency_single_point(sr, accuracy, alpha=1, beta=1):
    bpe = alpha * (1-sr**2) * beta * accuracy
    return bpe

def balanced_performance_efficiency_single_points(srs, accuracies, alpha=1, beta=1):
    bpes = []
    for i, sr in enumerate(srs):
        bpe = alpha * (1-sr**2) * beta * accuracies[i]
        bpes.append(bpe)
        
    return bpes

def balanced_performance_efficiency_multiple_points(srs, accuracies, alpha=1, beta=1):
    bpe_term = []
    normalization_term = []
    n = len(srs) - 1
    for i in range(n):
         bpe_area = (srs[i] - srs[i+1]) * (accuracies[i] * (1-srs[i]**2) + accuracies[i+1] * (1-srs[i+1]**2)) * 1/2 * alpha
         bpe_term.append(bpe_area)
         
         normalization_area = (srs[i] - srs[i+1]) * ((1-srs[i]**2) + (1-srs[i+1]**2)) * 1/2 * beta
         normalization_term.append(normalization_area)
         
    bpe = np.sum(bpe_term)
    bpe_normalized = bpe/np.sum(normalization_term)
    
    return bpe_normalized

# %% -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
from typing import List
import warnings, textwrap
from scipy import stats as st

def _apply_sr_ticks_and_vlines(ax: plt.Axes, sr_values, vline_kwargs: dict | None = None, tick_labels: List[str] | None = None):
    """
    - 将 x 轴刻度设为给定 sr 集合（去重后按降序）。
    - 在每个 sr 位置画竖直虚线（贯穿当前 y 轴范围）。
    """
    sr_unique = np.array(sorted(np.unique(sr_values), reverse=True), dtype=float)
    # 设刻度
    ax.set_xticks(sr_unique)
    if tick_labels is None:
        ax.set_xticklabels([str(s) for s in sr_unique], fontsize=14)
    else:
        ax.set_xticklabels(tick_labels, fontsize=14)

    # 先拿到绘完图后的 y 轴范围，再画竖线以贯穿全高
    y0, y1 = ax.get_ylim()
    kw = dict(color="gray", linestyle="--", linewidth=0.8, alpha=0.45, zorder=1)
    if vline_kwargs:
        kw.update(vline_kwargs)
    for x in sr_unique:
        ax.vlines(x, y0, y1, **kw)
    # 不改变 y 轴范围
    ax.set_ylim(y0, y1)

def compute_error_band(m, s, *,
                       mode: str = "ci", level: float = 0.95, n: int | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    mode = mode.lower()
    m = np.asarray(m, dtype=float)
    s = np.asarray(s, dtype=float)
    
    if mode == "none":
        delta, low, high, note = 0, m, m, None
        return delta, low, high, note
    
    # --- SD 模式 ---
    if mode == "sd":
        low, high = m - s, m + s
        return low, high, "±SD"

    # --- 需要 n ---
    if n is None:
        warnings.warn(f"[compute_error_band] n 未提供，{mode.upper()} 退化为 SD 阴影带（仅作展示）。")
        return m - s, m + s, "±SD (fallback)"

    n = np.asarray(n, dtype=float)
    sem = s / np.sqrt(n)

    # --- SEM 模式 ---
    if mode == "sem":
        low, high = m - sem, m + sem
        return low, high, f"±SEM (n={np.mean(n):.0f})"

    # --- CI 模式 ---
    dof = np.maximum(1, n - 1)
    # tcrit 支持广播（需逐点计算）
    tcrit = st.t.ppf((1.0 + level) / 2.0, df=dof)
    delta = tcrit * sem

    low, high = m - delta, m + delta
    note = f"±{int(level*100)}% CI (n={np.mean(n):.0f})"
    return delta, low, high, note

def plot_lines_with_band(df: pd.DataFrame, identifier: str = "identifier", iv: str = "srs", dv: str = "data", std: str = "stds",
    ylabel: str = "YLABEL", xlabel="XLABEL",
    mode: str = "ci", level: float = 0.95, n: int | None = None, 
    figsize=(10, 6), fontsize: int = 16, cmap=plt.colormaps['viridis'],
    use_alt_linestyles: bool = False,
    linestyles = None,
    facecolor: str = 'white',
    ) -> None: 
    # plot
    if cmap is None: 
        cmap = plt.colormaps['viridis']
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)

    grouped = list(df.groupby(identifier, sort=False))
    gcount = len(grouped)

    for i, (method, values) in enumerate(grouped):
        values = values.sort_values(iv, ascending=False)
        x = values[iv].to_numpy()   # iv
        m = values[dv].to_numpy()   # dv, magnitude
        s = values[std].to_numpy()  # std

        # Error Calculation
        _, low, high, band_note = compute_error_band(m, s, mode=mode, level=level, n=n)

        # 颜色
        color_value = i / max((gcount - 1), 1)

        # NEW: 按组索引 i 决定线型（False -> 全实线；True -> 实虚交替）
        if linestyles is not None:
            linestyle = linestyles[i]
        elif linestyles is None:
            linestyle = '--' if (use_alt_linestyles and (i % 2 == 1)) else '-'

        # Plot Lines + Error Bars
        ax.plot(x, m, marker="o", linewidth=2.0, label=method, zorder=3, color=cmap(color_value),
            linestyle=linestyle)
        ax.fill_between(x, low, high, alpha=0.15, zorder=2, color=cmap(color_value))

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.invert_xaxis()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_facecolor(facecolor)

    # 只按 SR 标定刻度，并在刻度处加竖线
    _apply_sr_ticks_and_vlines(ax, df[iv])

    ax.tick_params(axis="x", labelsize=fontsize * 0.9)
    ax.tick_params(axis="y", labelsize=fontsize * 0.9)

    legend = ax.legend(fontsize=fontsize * 0.9,
                       title=(f"Error Bands: {band_note}" if band_note is not None else ""),
                       title_fontsize=fontsize)
    legend.get_frame().set_facecolor(facecolor)
    
    fig.tight_layout()
    plt.show()

def plot_bars(df: pd.DataFrame, identifier: str = "identifier", iv: str = "srs", dv: str = "data", std: str = "stds",
    mode: str = "sem", level: float = 0.95, n: int | None = None, 
    ylabel: str = "YLABEL", xlabel: str = "XLABEL",
    error_handle_label = None,
    figsize = (10,10), lower_limit = 'auto', fontsize: int = 16, bar_width: float = 0.6, capsize: float = 5, 
    color_bar: str = "auto", bar_colors = None, cmap=plt.colormaps['viridis'],
    annotate: bool = True, annotate_fmt: str = "{m:.2f} ± {e:.2f}",
    xtick_rotation: float = 30, wrap_width: int | None = None,
    hatchs = None
    ) -> None:
    # 若 df 含重复 Method，则聚合
    df_preprocessed = df.groupby(identifier, sort=False).agg({dv: "mean", std: "mean"}).reset_index()
    
    # 提取方法与统计值
    methods = df_preprocessed[identifier].astype(str).tolist()

    # 自动换行
    if wrap_width is not None:
        methods_wrapped = [textwrap.fill(m, wrap_width) for m in methods]
    else:
        methods_wrapped = methods

    means = df_preprocessed[dv].to_numpy()
    stds = df_preprocessed[std].to_numpy()
    
    # Error Calculation
    errs, _, _, err_note = compute_error_band(means, stds, mode=mode, level=level, n=n)

    # 绘制
    num_methods = len(methods)
    x = np.arange(num_methods)
    
    if color_bar == "manual":
        if bar_colors is None: 
            bar_colors = ['skyblue'] * (num_methods-1) + ['orange']
    elif color_bar == "auto":
        bar_colors = []
        if cmap is None: 
            cmap=cm.get_cmap('viridis')
        for i in range(num_methods):
            color_value = i / max((num_methods - 1), 1)
            bar_colors.append(cmap(color_value))
        
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, means, width=bar_width,
                  yerr=errs, capsize=capsize,
                  color=bar_colors, edgecolor='black')
    
    if hatchs is not None:
        for bar, hatch in zip(bars, hatchs):
            bar.set_hatch(hatch)
            bar.set_edgecolor('white')
    
    # 注释数值
    if annotate:
        for xx, m, e in zip(x, means, errs):
            ax.text(xx, m + e + 0.3, annotate_fmt.format(m=m, e=e),
                    ha="center", va="bottom", fontsize=fontsize * 0.8)
        
    # 坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(methods_wrapped, fontsize=fontsize * 0.9,
                       rotation=xtick_rotation, ha="right" if xtick_rotation != 0 else "center")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis="y", labelsize=fontsize * 0.9)
    
    if lower_limit == 'auto':
        ymin = min(means) - max(errs) - 5
    elif isinstance(lower_limit, int):
        ymin = lower_limit
        
    ax.set_ylim(bottom=ymin)
    
    # 图例
    # ax.legend([], [f"Errors: {err_note}"], fontsize=fontsize * 0.8, title_fontsize=fontsize)
    
    # 创建一个 "H" 形误差棒图例符号
    if error_handle_label is None:
        error_handle_label=f'Errors {err_note}'
    else: 
        error_handle_label=error_handle_label
        
    error_handle = mlines.Line2D([], [], color='black',
                                 marker='_', markersize=3,   # 控制端帽长度
                                 markeredgewidth=10,          # 控制端帽线宽
                                 linestyle='-', linewidth=1.5,  # 中间竖线
                                 label=error_handle_label)
    
    ax.legend(handles=[error_handle], fontsize=fontsize * 0.8, title_fontsize=fontsize)    
    
    fig.tight_layout()
    plt.show()

# %% -----------------------------
def accuracy_partia(feature='pcc', model='basic', f1score=False):
    # color map
    cmap = plt.colormaps['tab20_r']
    
    # import data
    if feature == 'pcc':
        if model == 'basic':
            from results_summary import partia_data_pcc as partia_data
        elif model == 'advanced':
            from results_summary import partia_data_pcc_ad as partia_data
    elif feature == 'plv':
        if model == 'basic':
            from results_summary import partia_data_plv as partia_data
        elif model == 'advanced':
            from results_summary import partia_data_plv_ad as partia_data
    
    # accuracy
    accuracy_dic = partia_data.accuracy
    df_accuracy = pd.DataFrame(accuracy_dic)
    
    plot_lines_with_band(df_accuracy, dv='data', std='stds', 
                        mode="ci", n=30, 
                        ylabel="Average Accuracy (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, use_alt_linestyles=True)
    
    plot_lines_with_band(df_accuracy, dv='stds', std='stds', 
                        mode="none", 
                        ylabel="Accuracy Std. (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, use_alt_linestyles=True)
    
    # f1 score
    df_f1score=None
    if f1score:
        f1score_dic = partia_data.f1score
        df_f1score = pd.DataFrame(f1score_dic)
        
        plot_lines_with_band(df_f1score, dv='data', std='stds', 
                            mode="ci", n=30, 
                            ylabel="Average F1 Score (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                            cmap=cmap, use_alt_linestyles=True)
        
        plot_lines_with_band(df_f1score, dv='stds', std='stds', 
                            mode="none", 
                            ylabel="F1 Score Std. (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                            cmap=cmap, use_alt_linestyles=True)
    
    return df_accuracy, df_f1score

def sbpe_partia(feature='pcc', model='basic'):
    # color map
    cmap = plt.colormaps['tab20_r']
    
    # import data
    if feature == 'pcc':
        if model == 'basic':
            from results_summary import partia_data_pcc as partia_data
        elif model == 'advanced':
            from results_summary import partia_data_pcc_ad as partia_data
    elif feature == 'plv':
        if model == 'basic':
            from results_summary import partia_data_plv as partia_data
        elif model == 'advanced':
            from results_summary import partia_data_plv_ad as partia_data
    
    accuracy_dic = partia_data.accuracy
    df_accuracy = pd.DataFrame(accuracy_dic)
    
    sbpes, sbpe_stds = [], []
    for method, sub in df_accuracy.groupby("identifier", sort=False):
        sub = sub.sort_values("srs", ascending=False)
        srs = sub["srs"].to_numpy()
        accuracies = sub["data"].to_numpy()
        stds = sub["stds"].to_numpy()
        
        sbpes_ = balanced_performance_efficiency_single_points(srs, accuracies)
        sbpe_stds_ = balanced_performance_efficiency_single_points(srs, stds)
        
        # print(srs)
        # print(accuracies)
        # print(f"Methods: {method}", f"SBPEs: {sbpes_}")
        
        sbpes.extend(sbpes_)
        sbpe_stds.extend(sbpe_stds_)
        
    sbpes_dic = {"SBPEs": sbpes, "SBPE_stds": sbpe_stds}
    df_augmented = pd.concat([df_accuracy, pd.DataFrame(sbpes_dic)], axis=1)
    
    plot_lines_with_band(df_augmented, dv='SBPEs', std='SBPE_stds', 
                        mode="ci", n=30, 
                        ylabel='BPE (Balanced Performance Efficiency) (%)', xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, use_alt_linestyles=True)
    
    return df_augmented

def mbpe_partia(feature='pcc', model='basic'):
    # color map
    cmap = plt.colormaps['tab20_r']
    # hatchs
    hatchs = ['', '/', '', '/'] * 10
    
    # import data
    if feature == 'pcc':
        if model == 'basic':
            from results_summary import partia_data_pcc as partia_data
        elif model == 'advanced':
            from results_summary import partia_data_pcc_ad as partia_data
    elif feature == 'plv':
        if model == 'basic':
            from results_summary import partia_data_plv as partia_data
        elif model == 'advanced':
            from results_summary import partia_data_plv_ad as partia_data
    
    accuracy_dic = partia_data.accuracy
    df_accuracy = pd.DataFrame(accuracy_dic)
    
    mbpe, mbpe_std = [], []
    for method, sub in df_accuracy.groupby("identifier", sort=False):
        sub = sub.sort_values("srs", ascending=False)
        srs = sub["srs"].to_numpy()
        accuracies = sub["data"].to_numpy()
        stds = sub["stds"].to_numpy()
        
        mbpe_ = balanced_performance_efficiency_multiple_points(srs, accuracies)
        mbpe_std_ = balanced_performance_efficiency_multiple_points(srs, stds)
        
        # print(srs)
        # print(accuracies)
        print(f"Methods: {method}", f"MBPE: {mbpe_}")
        
        mbpe_ = [mbpe_] * len(accuracies)
        mbpe_std_ = [mbpe_std_] * len(accuracies)
        
        mbpe.extend(mbpe_)
        mbpe_std.extend(mbpe_std_)
    
    mbpe_dic = {"MBPEs": mbpe, "MBPE_stds": mbpe_std}
    df_augmented = pd.concat([df_accuracy, pd.DataFrame(mbpe_dic)], axis=1)

    plot_bars(df_augmented, dv="MBPEs", std="MBPE_stds", 
              mode="ci", n=30, 
              color_bar="auto", cmap=cmap,
              ylabel="BPE (Balanced Performance Efficiency) (%)", xlabel="FN Recovery Methods",
              xtick_rotation=30, wrap_width=30, figsize=(10,10), lower_limit=70, hatchs=hatchs)
    
    return df_augmented
    
# %% -----------------------------
def accuracy_selected():
    # color bars; plot settings
    from matplotlib.colors import ListedColormap
    
    cmap = ListedColormap([
        "lightsteelblue", # "#6A5ACD",  # PLI
        "limegreen", "green",  # PCC
        "salmon", "crimson"   # PLV
    ])
    
    linestyles = ['--', '--', '-', '--', '-']
    # end
    
    # data
    from results_summary import selected_data
    
    # accuracy
    accuracy_dic = selected_data.accuracy
    df_accuracy = pd.DataFrame(accuracy_dic)
    
    plot_lines_with_band(df_accuracy, dv='data', std='stds', 
                        mode="ci", n=30, 
                        ylabel="Average Accuracy (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, linestyles=linestyles)
    
    plot_lines_with_band(df_accuracy, dv='stds', std='stds', 
                        mode="none", n=1, 
                        ylabel="Accuracy Std. (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, linestyles=linestyles)
    
    # f1 score
    f1score_dic = selected_data.f1score
    df_f1score = pd.DataFrame(f1score_dic)
    
    plot_lines_with_band(df_f1score, dv='data', std='stds', 
                        mode="ci", n=30, 
                        ylabel="Average F1 Score (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, linestyles=linestyles)
    
    plot_lines_with_band(df_f1score, dv='stds', std='stds', 
                        mode="none", n=1, 
                        ylabel="F1 Score Std. (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, linestyles=linestyles)
    
    return df_accuracy, df_f1score

def accuracy_appendix(method='glf', feature='pcc'):
    # color bars; plot settings
    from matplotlib.colors import ListedColormap
    
    cmap = ListedColormap([
        "lightsteelblue", "blue",
        "limegreen", "salmon",
        "lightsteelblue", "blue",
        "limegreen", "salmon",
    ])
    
    linestyles = ['-', '-', '-', '-', '--', '--', '--', '--']
    # end
    
    # data
    if method == 'glf' and feature =='pcc':
        from results_appendix import glf_pcc as data
    elif method == 'glf' and feature == 'plv':
        from results_appendix import glf_plv as data
    
    # accuracy; pcc
    accuracy_dic = data.accuracy
    df_accuracy = pd.DataFrame(accuracy_dic)
    
    plot_lines_with_band(df_accuracy, dv='data', std='stds', 
                        mode="ci", n=30, 
                        ylabel="Average Accuracy (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, linestyles=linestyles)
    
    plot_lines_with_band(df_accuracy, dv='stds', std='stds', 
                        mode="none", n=1, 
                        ylabel="Accuracy Std. (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, linestyles=linestyles)
    
    # f1 score; pcc
    f1score_dic = data.f1score
    df_f1score = pd.DataFrame(f1score_dic)
    
    plot_lines_with_band(df_f1score, dv='data', std='stds', 
                        mode="ci", n=30, 
                        ylabel="Average F1 Score (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, linestyles=linestyles)
    
    plot_lines_with_band(df_f1score, dv='stds', std='stds', 
                        mode="none", n=1, 
                        ylabel="F1 Score Std. (%)", xlabel="Node Selection Rate (for Subnetwork Extraction)",
                        cmap=cmap, linestyles=linestyles)
    
    return df_accuracy, df_f1score

def mbpe_appendix(method='glf',feature='pcc'):
    # color bars; plot settings
    from matplotlib.colors import ListedColormap
    
    cmap = ListedColormap([
        "lightsteelblue", "blue",
        "limegreen", "salmon",
        "lightsteelblue", "blue",
        "limegreen", "salmon",
    ])
    # hatchs
    hatchs = ['', '', '', '', '/', '/', '/', '/']
    
    # import data
    if method == 'glf' and feature =='pcc':
        from results_appendix import glf_pcc as data
        xlabel="Parameter Settings of Graph Laplacian Filtering"
    elif method == 'glf' and feature == 'plv':
        from results_appendix import glf_plv as data
        xlabel="Parameter Settings of Graph Laplacian Filtering"
    
    accuracy_dic = data.accuracy
    df_accuracy = pd.DataFrame(accuracy_dic)
    
    mbpe, mbpe_std = [], []
    for method, sub in df_accuracy.groupby("identifier", sort=False):
        sub = sub.sort_values("srs", ascending=False)
        srs = sub["srs"].to_numpy()
        accuracies = sub["data"].to_numpy()
        stds = sub["stds"].to_numpy()
        
        mbpe_ = balanced_performance_efficiency_multiple_points(srs, accuracies)
        mbpe_std_ = balanced_performance_efficiency_multiple_points(srs, stds)
        
        # print(srs)
        # print(accuracies)
        print(f"Methods: {method}", f"MBPE: {mbpe_}")
        
        mbpe_ = [mbpe_] * len(accuracies)
        mbpe_std_ = [mbpe_std_] * len(accuracies)
        
        mbpe.extend(mbpe_)
        mbpe_std.extend(mbpe_std_)
    
    mbpe_dic = {"MBPEs": mbpe, "MBPE_stds": mbpe_std}
    df_augmented = pd.concat([df_accuracy, pd.DataFrame(mbpe_dic)], axis=1)

    plot_bars(df_augmented, dv="MBPEs", std="MBPE_stds", 
              mode="ci", n=30, 
              color_bar="auto", cmap=cmap,
              ylabel="BPE (Balanced Performance Efficiency) (%)", xlabel=xlabel,
              xtick_rotation=30, wrap_width=30, figsize=(10,10), lower_limit=70, hatchs=hatchs)
    
    return df_augmented

# %% main
if __name__ == "__main__":   
    # %% partia_basic
    accuracy_pcc, f1score_pcc = accuracy_partia('pcc')
    df_sbpe = sbpe_partia('pcc')
    df_mbpe = mbpe_partia('pcc')
    
    from results_plot_p_matrix import plot_basic_models_comparison
    plot_basic_models_comparison()
    
    # %% partia_advanced
    # accuracy_pcc, f1score_pcc = accuracy_partia('pcc', 'advanced')
    # df_sbpe = sbpe_partia('pcc', 'advanced')
    # df_mbpe = mbpe_partia('pcc', 'advanced')
    
    # from results_plot_p_matrix import plot_advanced_models_comparison
    # plot_advanced_models_comparison()
    
    # %% selected
    # acc, f1 = accuracy_selected()