# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:23:51 2025

@author: usouu
"""

import matplotlib.pyplot as plt

# X轴：通道保留率
rates = [1, 0.5, 0.25, 0.2]

# Y轴：每种方法在不同rate下的准确率
data = {
    "Label_Driven_MI": [81.28577621, 81.23708106, 79.76336012, 79.92137764],
    "Data_Driven_MI": [81.28577621, 75.32968663, 69.50397495, 68.46430621],
    "Data_Driven_PCC": [81.28577621, 81.46805384, 70.48138248, 69.0668316],
    "Data_Driven_PLV": [81.28577621, 75.15932952, 67.74383477, 63.8484589],
    "sigmoid_model_PCC": [81.28577621, 78.5806567, 77.4806491, 76.55120238],
    "g_gaussian_model_PCC": [81.28577621, 78.86768532, 74.72557529, 75.97405399],
}

# 绘图
plt.figure(figsize=(12, 8))
for method, accuracies in data.items():
    plt.plot(rates, accuracies, marker='o', label=method)

# 配置图表
plt.title("Accuracy vs. Rate of Channel Selection")
plt.xlabel("Rate of Channel Selection")
plt.ylabel("Accuracy (%)")
plt.xticks(rates)
plt.gca().invert_xaxis()  # 反转X轴
plt.grid(True)
plt.legend()
plt.tight_layout()

# 显示图形
plt.show()
