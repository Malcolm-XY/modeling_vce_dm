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

# effect-values
def plot_diff_e_heatmap( 
    effect_df: pd.DataFrame,   # Cohen's d
    diff_df: pd.DataFrame,     # Δmean (row - col)
    title: str | None = None,
    cmap_diff: str = "coolwarm",   # 上三角色图
    cmap_e: str = "Purples",       # 下三角色图
    fmt: str = ".2f",
    same_cmap: bool = False,
    draw_diagonal: bool = True
):
    import textwrap

    # —— 文本换行 ——
    def wrap_labels(labels, width=25):
        return ['\n'.join(textwrap.wrap(l, width)) for l in labels]

    # —— 若要统一颜色 —— 
    if same_cmap:
        cmap_e = cmap_diff

    # —— 对齐 —— 
    diff_df = diff_df.loc[effect_df.index, effect_df.columns]
    effect_df = effect_df.loc[effect_df.index, effect_df.columns]
    methods = diff_df.index.tolist()
    k = len(methods)

    # —— 建立画布布局 —— 
    fig = plt.figure(figsize=(10, 8))
    
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
    diff_vals = diff_df.values.astype(float)
    eff_vals = effect_df.values.astype(float)

    diff_max = np.nanmax(np.abs(diff_vals))
    vmin_diff, vmax_diff = -diff_max, diff_max

    eff_max = np.nanmax(np.abs(eff_vals))
    vmin_eff, vmax_eff = 0.0, eff_max

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
        cbar_kws={"label": "Δ mean (row - col)"}
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
        cbar_kws={"label": "Effect size |d| (col - row)", 
                  "orientation": "horizontal"}
    )

    # —— 文本标注 —— 
    annot = np.empty((k, k), dtype=object)

    for i in range(k):
        for j in range(k):
            if i == j:
                annot[i, j] = ""
            elif i < j:
                # 上三角：行 − 列（Δmean）
                d = diff_df.iloc[i, j]
                arrow = "↑" if d > 0 else ("↓" if d < 0 else "")
                annot[i, j] = f"{d:{fmt}}{arrow}"
            else:
                # 下三角：列 − 行（效应量原符号）
                e = effect_df.iloc[i, j]
                annot[i, j] = f"{e:{fmt}}"

    # —— 根据背景色调整字体颜色 —— 
    cm_diff = plt.cm.get_cmap(cmap_diff)
    cm_eff = plt.cm.get_cmap(cmap_e)

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
        
    plt.tight_layout()
    plt.show()

def estimate_effect_matrix_from_summary(
    df: pd.DataFrame,
    identifier: str = "identifier",
    iv: str = "srs",
    mean_col: str = "data",
    std_col: str = "stds",
    n: int = 30,
):
    effect_matrices: dict[float, pd.DataFrame] = {}
    diff_matrices: dict[float, pd.DataFrame] = {}

    for iv_value, sub in df.groupby(iv):

        sub = sub.set_index(identifier)
        methods = list(sub.index)

        means = sub[mean_col].astype(float)
        stds = sub[std_col].astype(float)

        k = len(methods)

        # 效应量矩阵 (Cohen's d)
        d_mat = np.zeros((k, k), dtype=float)

        # 均值差矩阵 Δm = m_i - m_j
        diff_mat = np.zeros((k, k), dtype=float)

        for i in range(k):
            for j in range(i + 1, k):

                m1, m2 = means.iloc[i], means.iloc[j]
                s1, s2 = stds.iloc[i], stds.iloc[j]

                # 均值差
                diff = m1 - m2
                diff_mat[i, j] = diff
                diff_mat[j, i] = -diff

                # pooled standard deviation (equal n assumed)
                # s_p = sqrt( ((n-1)*s1^2 + (n-1)*s2^2) / (2n-2) )
                denom = (2 * n - 2)
                if denom <= 0:
                    d = np.nan
                else:
                    sp2 = ((n - 1) * s1**2 + (n - 1) * s2**2) / denom
                    sp = np.sqrt(sp2)

                    if sp == 0:
                        d = np.nan
                    else:
                        d = diff / sp

                d_mat[i, j] = d
                d_mat[j, i] = -d  # 反号

        d_df = pd.DataFrame(d_mat, index=methods, columns=methods)
        diff_df = pd.DataFrame(diff_mat, index=methods, columns=methods)

        effect_matrices[iv_value] = d_df
        diff_matrices[iv_value] = diff_df

    return effect_matrices, diff_matrices

# p-values
from scipy.stats import t as student_t
def estimate_p_matrix_from_summary(
    df: pd.DataFrame,
    identifier: str = "identifier",
    iv: str = "srs",
    mean_col: str = "data",
    std_col: str = "stds",
    n: int = 30,
    two_tailed: bool = True,
):
    p_matrices: dict[float, pd.DataFrame] = {}
    diff_matrices: dict[float, pd.DataFrame] = {}

    for iv_value, sub in df.groupby(iv):

        sub = sub.set_index(identifier)
        methods = list(sub.index)

        means = sub[mean_col].astype(float)
        stds = sub[std_col].astype(float)

        k = len(methods)

        # p 矩阵
        p_mat = np.full((k, k), np.nan, dtype=float)

        # 均值差矩阵 Δm = m_i - m_j
        diff_mat = np.zeros((k, k), dtype=float)

        for i in range(k):
            for j in range(i + 1, k):

                m1, m2 = means.iloc[i], means.iloc[j]
                s1, s2 = stds.iloc[i], stds.iloc[j]

                # 均值差
                diff = m1 - m2
                diff_mat[i, j] = diff
                diff_mat[j, i] = -diff

                # 标准误差
                se = np.sqrt(s1**2 / n + s2**2 / n)
                if se == 0:
                    p = np.nan
                else:
                    t_stat = diff / se

                    # Welch–Satterthwaite df
                    v1 = s1**2 / n
                    v2 = s2**2 / n
                    df_welch = (v1 + v2)**2 / (
                        (v1**2)/(n - 1) + (v2**2)/(n - 1)
                    )

                    # p-value
                    if two_tailed:
                        p = 2 * (1 - student_t.cdf(abs(t_stat), df_welch))
                    else:
                        p = 1 - student_t.cdf(abs(t_stat), df_welch)

                p_mat[i, j] = p
                p_mat[j, i] = p

        p_df = pd.DataFrame(p_mat, index=methods, columns=methods)
        diff_df = pd.DataFrame(diff_mat, index=methods, columns=methods)

        p_matrices[iv_value] = p_df
        diff_matrices[iv_value] = diff_df

    return p_matrices, diff_matrices

import seaborn as sns
def plot_diff_p_heatmap(
    p_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    title: str | None = None,
    cmap_diff: str = "coolwarm",   # 均值差着色
    cmap_p: str = "Purples",       # 显著性着色（可与均值差一致）
    center: float = 0.0,
    fmt: str = ".2f",
):
    import textwrap

    def wrap_labels(labels, width=25):
        return ['\n'.join(textwrap.wrap(l, width)) for l in labels]

    # -----------------------------
    # Align
    # -----------------------------
    diff_df = diff_df.loc[p_df.index, p_df.columns]
    methods = wrap_labels(p_df.index.tolist(), width=30)
    k = len(methods)

    # -----------------------------
    # Build color matrix
    # -----------------------------
    diff_for_color = np.zeros((k, k), dtype=float)

    # 上三角：diff
    for i in range(k):
        for j in range(k):
            if i < j:
                diff_for_color[i, j] = diff_df.iloc[i, j]
            elif i == j:
                diff_for_color[i, j] = np.nan
            else:
                # 下三角：由 p 值等级决定颜色
                p = p_df.iloc[j, i]
                if p < 0.001:
                    level = 3
                elif p < 0.01:
                    level = 2
                elif p < 0.05:
                    level = 1
                else:
                    level = 0
                diff_for_color[i, j] = level

    # -----------------------------
    # Build annot
    # -----------------------------
    annot = np.empty((k, k), dtype=object)
    annot[:, :] = ""

    def p_to_stars(p):
        if p < 0.001:  return "***"
        if p < 0.01:   return "**"
        if p < 0.05:   return "*"
        return ""

    for i in range(k):
        for j in range(k):
            if i == j:
                annot[i, j] = ""
            elif i < j:
                diff = diff_df.iloc[i, j]
                arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "")
                annot[i, j] = f"{diff:{fmt}}{arrow}"
            else:
                # p-value level + trend direction
                diff = diff_df.iloc[j, i]  # 对称
                arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "")
                p = p_df.iloc[j, i]
                stars = p_to_stars(p)
                annot[i, j] = f"{stars}{arrow}"

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(0.8 * k + 3, 0.8 * k + 3))

    # 使用两个 colormap：
    #   - 上三角 diff：centered cmap_diff
    #   - 下三角 p：normalized cmap_p
    # 解决办法：用 ListedColormap 合并两套色图
    import matplotlib.colors as mcolors

    # 创建显著性 colormap：从 cmap_p 取 4 个色块
    p_levels = 4
    cmap_p_obj = plt.get_cmap(cmap_p, p_levels)

    # 创建 diff colormap
    cmap_diff_obj = plt.get_cmap(cmap_diff)

    # 构建一个组合 colormap（以便 heatmap 一次画完）
    # 假设 diff 范围远大于 p-level 范围，不冲突即可
    # diff: float, p-level: small int
    cmap_combined = cmap_diff_obj

    ax = sns.heatmap(
        diff_for_color,
        annot=annot,
        fmt="",
        cmap=cmap_combined,
        center=center,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        xticklabels=methods,
        yticklabels=methods,
        cbar=False,
        cbar_kws={"shrink": 0.65, "aspect": 30}
    )

    # 对角线框
    for i in range(k):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=1))

    # ① 把 x 轴刻度放上面
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", top=True, bottom=False)
    ax.xaxis.set_label_position("top")
    
    # ② 调整 ticklabel 的位置（关键）
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="left", 
             va="bottom")    # 垂直对齐底部 → 避免移位
    
    # ③ 增加 padding（tick 到轴之间的距离）
    ax.tick_params(axis="x", pad=5) 

    plt.yticks(rotation=0)
    
    if title is not None:
        plt.title(title, pad=20)

    plt.tight_layout()
    plt.show()

    return annot

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
    
    # perparation
    df_simple = df_mbpe.groupby('identifier', sort=False).first().reset_index()
    r1, r2, r3, r4, r5, r6 = df_simple.loc[2], df_simple.loc[3], df_simple.loc[4], df_simple.loc[5], df_simple.loc[1], df_simple.loc[0]
    df = pd.DataFrame([r1,r2,r3,r4,r5,r6])
    
    # p values and matrix
    p_matrix, diff_matrix = estimate_p_matrix_from_summary(df, mean_col='MBPEs', std_col='MBPE_stds', n=30)
    ann = plot_diff_p_heatmap(p_matrix[1], diff_matrix[1])
    
    # effect values
    e_matrix, diff_matrix = estimate_effect_matrix_from_summary(df, mean_col='MBPEs', std_col='MBPE_stds', n=30)
    ann = plot_diff_e_heatmap(-e_matrix[1], diff_matrix[1])
    
    # %% partia_advanced
    # accuracy_pcc, f1score_pcc = accuracy_partia('pcc', 'advanced')
    # df_sbpe = sbpe_partia('pcc', 'advanced')
    # df_mbpe = mbpe_partia('pcc', 'advanced')
    
    # # p values and matrix
    # df_simple = df_mbpe.groupby('identifier', sort=False).first().reset_index()
    # r1, r2, r3, r4, r5, r6 = df_simple.loc[2], df_simple.loc[3], df_simple.loc[4], df_simple.loc[5], df_simple.loc[1], df_simple.loc[0]
    # df = pd.DataFrame([r1,r2,r3,r4,r5,r6])
    
    # p_matrix, diff_matrix = estimate_p_matrix_from_summary(df, mean_col='MBPEs', std_col='MBPE_stds', n=30)
    # ann = plot_diff_p_heatmap(p_matrix[1], diff_matrix[1])
    
    # # effect values
    # e_matrix, diff_matrix = estimate_effect_matrix_from_summary(df, mean_col='MBPEs', std_col='MBPE_stds', n=30)
    # ann = plot_diff_e_heatmap(-e_matrix[1], diff_matrix[1])
    
    # %% selected
    # acc, f1 = accuracy_selected()