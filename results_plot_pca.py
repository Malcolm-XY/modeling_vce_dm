# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 13:50:52 2025

@author: usouu
"""

import ci_management

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ====== 数据准备 ======
# 假设 ci_management.read_channel_importances_fitted(...)['ams'] 返回 np.ndarray 或 list
ci_target = ci_management.read_channel_importances_fitted('advanced', 'linear_ratio', 'target')['ams']
ci_unfitted = ci_management.read_channel_importances_fitted('advanced', 'linear_ratio', 'non_modeled')['ams']
ci_fitted = {
    'exponential': ci_management.read_channel_importances_fitted('advanced', 'linear_ratio', 'exponential')['ams'],
    'gaussian': ci_management.read_channel_importances_fitted('advanced', 'linear_ratio', 'gaussian')['ams'],
    'inverse': ci_management.read_channel_importances_fitted('advanced', 'linear_ratio', 'inverse')['ams'],
    'powerlaw': ci_management.read_channel_importances_fitted('advanced', 'linear_ratio', 'powerlaw')['ams'],
    'rational_quadratic': ci_management.read_channel_importances_fitted('advanced', 'linear_ratio', 'rational_quadratic')['ams'],
    'generalized_gaussian': ci_management.read_channel_importances_fitted('advanced', 'linear_ratio', 'generalized_gaussian')['ams'],
    'sigmoid': ci_management.read_channel_importances_fitted('advanced', 'linear_ratio', 'sigmoid')['ams']
}

# 合并为一个 DataFrame
ci_dict = {'target': ci_target, 'non_modeled': ci_unfitted}
ci_dict.update(ci_fitted)
models = list(ci_dict.keys())
data = np.vstack([ci_dict[m] for m in models])  # shape: (n_models, n_channels)

# ====== PCA 降维 ======
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# ====== t-SNE 降维 ======
tsne = TSNE(n_components=2, perplexity=2, random_state=42, n_iter=2000)
tsne_result = tsne.fit_transform(data)

# ====== 绘图 ======
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- PCA ---
axes[0].scatter(pca_result[:, 0], pca_result[:, 1], s=80)
for i, m in enumerate(models):
    axes[0].text(pca_result[i, 0] + 0.01, pca_result[i, 1], m, fontsize=10)
axes[0].set_title(f'PCA Projection (PC1 var={pca.explained_variance_ratio_[0]:.2f}, PC2 var={pca.explained_variance_ratio_[1]:.2f})')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].grid(True, linestyle='--', alpha=0.3)

# --- t-SNE ---
axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], s=80)
for i, m in enumerate(models):
    axes[1].text(tsne_result[i, 0] + 0.01, tsne_result[i, 1], m, fontsize=10)
axes[1].set_title('t-SNE Projection (2D)')
axes[1].set_xlabel('Dim 1')
axes[1].set_ylabel('Dim 2')
axes[1].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()