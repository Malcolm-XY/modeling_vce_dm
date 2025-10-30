# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:03:59 2025

@author: usouu
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Updated data with four types of distances
data2 = {
    "alpha band": [0.403616051, 0.843077917, 0.362430502, 0.828703638],
    "beta band": [0.378883352, 0.839462445, 0.339527862, 0.821832319],
    "gamma band": [0.298452299, 0.817123989, 0.258022771, 0.796063128],
}
# index = [
#     "euclidean distance matrix",
#     "residual euclidean distance matrix",
#     "spherical distance matrix",
#     "residual spherical distance matrix",
# ]
index = [
    "DM (Euclidean)",
    "RDM (Euclidean)",
    "DM (Spherical)",
    "RDM (Spherical)",
]
df2 = pd.DataFrame(data2, index=index)

# Plot heatmap
plt.figure(figsize=(7, 4))
sns.heatmap(df2, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
plt.title("Cosine Similarity vs Distance Matrices")
plt.xlabel("Functional Network")
plt.ylabel("Distance Matrix Type")
plt.tight_layout()
plt.show()
