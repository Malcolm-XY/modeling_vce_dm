# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 22:05:36 2025

@author: 18307
"""

import numpy as np
import pandas as pd
def read_data(data_dict):
    df = pd.DataFrame({
    'data': data_dict['data'],
    'srs': data_dict['srs']
    })
    group_size = df['srs'].nunique()

    df['Group'] = np.arange(len(df)) // group_size
    df_pivot = df.pivot(index='Group', columns='srs', values='data')
    df_pivot_reversed = df_pivot.sort_index(axis=1, ascending=False)
    
    return df_pivot, df_pivot_reversed

# %% MBPE
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

def mbpe_for_data(data_dict):
    _, df = read_data(data_dict)
    
    mbpe = []
    for index, group in df.iterrows():
        srs_ = group.keys()
        data_ = group.tolist()
        
        mbpe_ = balanced_performance_efficiency_multiple_points(srs_, data_)
        mbpe.append(mbpe_)
        
    mbpe_averaged = np.mean(mbpe)
    
    return mbpe_averaged, mbpe

# %% P-Matrices
from scipy.stats import ttest_ind, ttest_rel
def compare_methods(data, alpha=0.05):
    methods = list(data.keys())
    n = len(methods)

    # 初始化结果矩阵
    mean_diff      = np.zeros((n, n))
    relative_gain  = np.zeros((n, n))
    p_matrix       = np.zeros((n, n))
    paired_p_matrix = np.ones((n, n))
    effect_size    = np.zeros((n, n))

    # 计算均值、标准差
    means = {m: np.mean(data[m]) for m in methods}
    stds  = {m: np.std(data[m], ddof=1) for m in methods}

    for i in range(n):
        for j in range(n):
            m1, m2 = methods[i], methods[j]
            x = np.array(data[m1])
            y = np.array(data[m2])

            # 1. 均值差
            mean_diff[i, j] = means[m1] - means[m2]

            # 2. 相对提升
            relative_gain[i, j] = (means[m1] - means[m2]) / (means[m2] + 1e-12)

            # 3. t 检验结果 (根据 paired 参数选择独立样本或配对样本)
            stat, p = ttest_ind(x, y, equal_var=False)
            p_matrix[i, j] = p

            # 4. paired t-test
            stat_paired, p_paired = ttest_rel(x, y)
            paired_p_matrix[i, j] = p_paired   # 不再是 0/1，而是配对 t 检验的 p 值

            # 5. 效应量（Cohen's d）
            pooled_std = np.sqrt((stds[m1]**2 + stds[m2]**2) / 2)
            effect_size[i, j] = (means[m1] - means[m2]) / (pooled_std + 1e-12)

    # 转成 DataFrame
    df_mean_diff      = pd.DataFrame(mean_diff,     index=methods, columns=methods)
    df_relative_gain  = pd.DataFrame(relative_gain, index=methods, columns=methods)
    df_p_matrix       = pd.DataFrame(p_matrix,      index=methods, columns=methods)
    df_paired_p_matrix = pd.DataFrame(paired_p_matrix, index=methods, columns=methods)
    df_effect_size    = pd.DataFrame(effect_size,   index=methods, columns=methods)

    return df_mean_diff, df_relative_gain, df_p_matrix, df_paired_p_matrix, df_effect_size

# %% Plot
import matplotlib.pyplot as plt
import seaborn as sns
def plot_rel_e_heatmap( 
    effect_df: pd.DataFrame,   # Cohen's d
    rel_df: pd.DataFrame,     #  (row / col)
    title: str | None = None,
    cmap_rel: str = "coolwarm",   # 上三角色图
    cmap_e: str = "Purples",       # 下三角色图
    fmt_rel: str = ".2%",
    fmt_e: str = ".2f",
    same_cmap: bool = False,
    draw_diagonal: bool = True
):
    import textwrap

    # —— 文本换行 ——
    def wrap_labels(labels, width=25):
        return ['\n'.join(textwrap.wrap(l, width)) for l in labels]

    # —— 若要统一颜色 —— 
    if same_cmap:
        cmap_e = cmap_rel

    # —— 对齐 —— 
    rel_df = rel_df.loc[effect_df.index, effect_df.columns]
    effect_df = effect_df.loc[effect_df.index, effect_df.columns]
    methods = rel_df.index.tolist()
    k = len(methods)

    # —— 建立画布布局 —— 
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    
    # 调整 gridspec 定义
    gs = fig.add_gridspec(
        nrows=5, ncols=5,
        width_ratios=[30, 30, 1, 1, 1], 
        height_ratios=[20, 35, 1, 1, 1],
        # 调整 wspace 和 hspace (如果需要)
        wspace=0.1, hspace=0.1
    )
    
    # 1. 主图 Axes (ax)
    ax = fig.add_subplot(gs[1, 1])
    # 2. 第一个 Colorbar (cax_diff) - 垂直
    cax_diff = fig.add_subplot(gs[1, 2])
    # 3. 第二个 Colorbar (cax_eff) - 水平，长度缩小
    cax_eff = fig.add_subplot(gs[2, 1])
    # ------
    
    
    # —— 颜色范围 —— 
    rel_vals = rel_df.values.astype(float)
    eff_vals = effect_df.values.astype(float)

    rel_max = np.nanmax(np.abs(rel_vals))
    vmin_rel, vmax_rel = -rel_max, rel_max

    eff_max = np.nanmax(np.abs(eff_vals))
    vmin_eff, vmax_eff = 0.0, eff_max

    # —— 上三角 ——
    mask_lower = np.tril(np.ones_like(rel_vals, dtype=bool), 0)
    sns.heatmap(
        rel_df,
        mask=mask_lower,
        cmap=cmap_rel,
        vmin=vmin_rel, vmax=vmax_rel,
        center=0.0,
        square=True,
        ax=ax,
        cbar=True,
        cbar_ax=cax_diff,
        cbar_kws={"label": "Relative Increase (row / col)"}
    )

    # —— 下三角（effect size）——
    mask_upper = np.triu(np.ones_like(eff_vals, dtype=bool), 0)
    sns.heatmap(
        np.abs(effect_df),  # 着色用绝对值，符号靠文本
        mask=mask_upper,
        cmap=cmap_e,
        vmin=vmin_eff, vmax=vmax_eff,
        square=True,
        ax=ax,
        cbar=True,
        cbar_ax=cax_eff,
        cbar_kws={"label": "Effect Size (col − row)",
                  "orientation": "horizontal"}
    )

    # —— 文本标注 —— 
    annot = np.empty((k, k), dtype=object)

    for i in range(k):
        for j in range(k):
            if i == j:
                annot[i, j] = ""
            elif i < j:
                # 上三角：行 − 列
                r = rel_df.iloc[i, j]
                arrow = "↑" if r > 0 else ("↓" if r < 0 else "")
                annot[i, j] = f"{r:{fmt_rel}}{arrow}"
            else:
                # 下三角：列 − 行（效应量原符号）
                e = effect_df.iloc[i, j]
                annot[i, j] = f"{e:{fmt_e}}"

    # —— 根据背景色调整字体颜色 —— 
    cm_diff = plt.cm.get_cmap(cmap_rel)
    cm_eff = plt.cm.get_cmap(cmap_e)

    for i in range(k):
        for j in range(k):

            text = annot[i, j]
            if text == "":
                continue

            # 计算背景颜色
            if i < j:
                # 上三角 → diff 用 cmap_diff
                val = rel_df.iloc[i, j]
                norm = (val - vmin_rel) / (vmax_rel - vmin_rel)
                r, g, b, _ = cm_diff(norm)
            else:
                # 下三角 → effect 用 cmap_e
                val = abs(effect_df.iloc[i, j])
                norm = (val - vmin_eff) / (vmax_eff - vmin_eff)
                r, g, b, _ = cm_eff(norm)

            # luminance
            lum = 0.299*r + 0.587*g + 0.114*b
            font_color = "white" if lum < 0.5 else "black"

            ax.text(
                j + 0.5, i + 0.5,
                text,
                ha="center", va="center",
                fontsize=9,
                color=font_color
            )

    # —— 刻度 —— 
    wrapped = wrap_labels(methods, width=25)

    ax.set_xticks(np.arange(k) + 0.5)
    ax.set_xticklabels(wrapped, rotation=35, ha="left")
    ax.set_yticks(np.arange(k) + 0.5)
    ax.set_yticklabels(wrapped)

    ax.xaxis.tick_top()
    ax.tick_params(axis='x', bottom=False, top=True,
                   labelbottom=False, labeltop=True)

    if title:
        ax.set_title(title, pad=40, fontsize=14)
        
    if draw_diagonal:
        for i in range(k):
            ax.add_line(plt.Line2D(
                [i, i+1], [i, i+1],
                color="black", linewidth=1.5, zorder=10
            ))

    plt.show()

def plot_diff_p_heatmap( 
    p_df: pd.DataFrame,   # Cohen's d
    diff_df: pd.DataFrame,     #  (row - col)
    title: str | None = None,
    cmap_diff: str = "coolwarm",   # 上三角色图
    cmap_p: str = "Purples",       # 下三角色图
    fmt_diff: str = ".2f",
    fmt_p: str = ".2f",
    same_cmap: bool = False,
    draw_diagonal: bool = True
):
    import textwrap

    # —— 文本换行 ——
    def wrap_labels(labels, width=25):
        return ['\n'.join(textwrap.wrap(l, width)) for l in labels]

    # —— 若要统一颜色 —— 
    if same_cmap:
        cmap_p = cmap_diff

    # —— 对齐 —— 
    diff_df = diff_df.loc[p_df.index, p_df.columns]
    p_df = p_df.loc[p_df.index, p_df.columns]
    methods = diff_df.index.tolist()
    k = len(methods)

    # —— 建立画布布局 —— 
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    
    # 调整 gridspec 定义
    gs = fig.add_gridspec(
        nrows=5, ncols=5,
        width_ratios=[30, 30, 1, 1, 1], 
        height_ratios=[20, 35, 1, 1, 1],
        # 调整 wspace 和 hspace (如果需要)
        wspace=0.1, hspace=0.1
    )
    
    # 1. 主图 Axes (ax)
    ax = fig.add_subplot(gs[1, 1])
    # 2. 第一个 Colorbar (cax_diff) - 垂直
    cax_diff = fig.add_subplot(gs[1, 2])
    # 3. 第二个 Colorbar (cax_eff) - 水平，长度缩小
    cax_p = fig.add_subplot(gs[2, 1])
    # ------
    
    # —— 颜色范围 —— 
    diff_vals = diff_df.values.astype(float)
    p_vals = p_df.values.astype(float)

    diff_max = np.nanmax(np.abs(diff_vals))
    vmin_diff, vmax_diff = -diff_max, diff_max

    p_max = np.nanmax(np.abs(p_vals))
    vmin_p, vmax_p = 0.0, p_max

    # —— 上三角（Δmean）——
    mask_lower = np.tril(np.ones_like(diff_vals, dtype=bool), 0)
    sns.heatmap(
        diff_df,
        mask=mask_lower,
        cmap=cmap_diff,
        vmin=vmin_diff, vmax=vmax_diff,
        center=0.0,
        square=True,
        ax=ax,
        cbar=True,
        cbar_ax=cax_diff,
        cbar_kws={"label": "Mean Difference Δ (row − col)"}
    )

    # —— 下三角 ——
    mask_upper = np.triu(np.ones_like(p_vals, dtype=bool), 0)
    sns.heatmap(
        np.abs(p_df),
        mask=mask_upper,
        cmap=cmap_p + '_r',
        vmin=vmin_p, vmax=vmax_p,
        square=True,
        ax=ax,
        cbar=True,
        cbar_ax=cax_p,
        cbar_kws={"label": "p-Value (Significance)",
                  "orientation": "horizontal"}
    )

    # —— 文本标注 —— 
    def p_to_stars(p):
        if p < 0.001:  return "***"
        if p < 0.01:   return "**"
        if p < 0.05:   return "*"
        return ""
    
    annot = np.empty((k, k), dtype=object)

    for i in range(k):
        for j in range(k):
            if i == j:
                annot[i, j] = ""
            elif i < j:
                # 上三角：行 − 列（Δmean）
                r = diff_df.iloc[i, j]
                arrow = "↑" if r > 0 else ("↓" if r < 0 else "")
                annot[i, j] = f"{r:{fmt_diff}}{arrow}"
            else:
                p = p_to_stars(p_df.iloc[i, j])
                annot[i, j] = f"{p}"# f"{p:{fmt_p}}"

    # —— 根据背景色调整字体颜色 —— 
    cm_diff = plt.cm.get_cmap(cmap_diff)
    cm_eff = plt.cm.get_cmap(cmap_p+'_r')

    for i in range(k):
        for j in range(k):

            text = annot[i, j]
            if text == "":
                continue

            # 计算背景颜色
            if i < j:
                # 上三角 → diff 用 cmap_diff
                val = diff_df.iloc[i, j]
                norm = (val - vmin_diff) / (vmax_diff - vmin_diff)
                r, g, b, _ = cm_diff(norm)
            else:
                # 下三角 → p 用 cmap_p
                val = abs(p_df.iloc[i, j])
                norm = (val - vmin_p) / (vmax_p - vmin_p)
                r, g, b, _ = cm_eff(norm)

            # luminance
            lum = 0.299*r + 0.587*g + 0.114*b
            font_color = "white" if lum < 0.5 else "black"

            ax.text(
                j + 0.5, i + 0.5,
                text,
                ha="center", va="center",
                fontsize=9,
                color=font_color
            )

    # —— 刻度 —— 
    wrapped = wrap_labels(methods, width=25)

    ax.set_xticks(np.arange(k) + 0.5)
    ax.set_xticklabels(wrapped, rotation=35, ha="left")
    ax.set_yticks(np.arange(k) + 0.5)
    ax.set_yticklabels(wrapped)

    ax.xaxis.tick_top()
    ax.tick_params(axis='x', bottom=False, top=True,
                   labelbottom=False, labeltop=True)

    if title:
        ax.set_title(title, pad=40, fontsize=14)
        
    if draw_diagonal:
        for i in range(k):
            ax.add_line(plt.Line2D(
                [i, i+1], [i, i+1],
                color="black", linewidth=1.5, zorder=10
            ))

    plt.show()

# %% Basic Model
def plot_basic_models_comparison():
    from results_append import accuracy_original as bl
    from results_append import accuracy_bs_generalized_surface_laplacian as gsf
    
    from results_append import accuracy_bs_residual_exponential as exp
    from results_append import accuracy_bs_residual_gaussian as gaus
    from results_append import accuracy_bs_residual_power_law as pl
    from results_append import accuracy_bs_residual_rational_quadratic as rq
    
    i_bl = bl['identifier']
    mbpe_ave_bl, mbpe_bl = mbpe_for_data(bl)
    i_glf = gsf['identifier']
    mbpe_ave_gsf, mbpe_gsf = mbpe_for_data(gsf)
    
    i_exp = exp['identifier']
    mbpe_ave_exp, mbpe_exp = mbpe_for_data(exp)
    i_gaus = gaus['identifier']
    mbpe_ave_gaus, mbpe_gaus = mbpe_for_data(gaus)
    
    i_pl = pl['identifier']
    mbpe_ave_pl, mbpe_pl = mbpe_for_data(pl)
    i_rq = rq['identifier']
    mbpe_ave_rq, mbpe_rq = mbpe_for_data(rq)
    
    data = {
        i_exp: mbpe_exp,
        i_gaus: mbpe_gaus,
        i_pl: mbpe_pl,
        i_rq: mbpe_rq,
        
        i_glf: mbpe_gsf,
        i_bl: mbpe_bl,
        }
    
    df_mean_diff, df_relative_gain, df_p_matrix, df_paired_p_matrix, df_effect_size = compare_methods(data)
    
    plot_rel_e_heatmap(-df_effect_size, df_relative_gain)
    plot_diff_p_heatmap(df_paired_p_matrix, df_mean_diff)

def plot_advanced_models_comparison():
    from results_append import accuracy_original as bl
    from results_append import accuracy_ad_generalized_surface_laplacian as gsf
    
    from results_append import accuracy_ad_residual_exponential as exp
    from results_append import accuracy_ad_residual_gaussian as gaus
    from results_append import accuracy_ad_residual_power_law as pl
    from results_append import accuracy_ad_residual_rational_quadratic as rq
    
    i_bl = bl['identifier']
    mbpe_ave_bl, mbpe_bl = mbpe_for_data(bl)
    i_glf = gsf['identifier']
    mbpe_ave_gsf, mbpe_gsf = mbpe_for_data(gsf)
    
    i_exp = exp['identifier']
    mbpe_ave_exp, mbpe_exp = mbpe_for_data(exp)
    i_gaus = gaus['identifier']
    mbpe_ave_gaus, mbpe_gaus = mbpe_for_data(gaus)
    
    i_pl = pl['identifier']
    mbpe_ave_pl, mbpe_pl = mbpe_for_data(pl)
    i_rq = rq['identifier']
    mbpe_ave_rq, mbpe_rq = mbpe_for_data(rq)
    
    data = {
        i_exp: mbpe_exp,
        i_gaus: mbpe_gaus,
        i_pl: mbpe_pl,
        i_rq: mbpe_rq,
        
        i_glf: mbpe_gsf,
        i_bl: mbpe_bl,
        }
    
    df_mean_diff, df_relative_gain, df_p_matrix, df_paired_p_matrix, df_effect_size = compare_methods(data)
    
    plot_rel_e_heatmap(-df_effect_size, df_relative_gain)
    plot_diff_p_heatmap(df_paired_p_matrix, df_mean_diff)

# %% main
if __name__ == "__main__":    
    plot_basic_models_comparison()
    # plot_advanced_models_comparison()