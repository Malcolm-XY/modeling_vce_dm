# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:32:18 2025

@author: 18307
"""
import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score

def train_and_evaluate_svm(X_train, Y_train, X_val, Y_val):
    model = SVC(kernel='rbf', C=1, gamma='scale')
    model.fit(X_train, Y_train)
    
    val_preds = model.predict(X_val)
    accuracy = accuracy_score(Y_val, val_preds) * 100
    recall = recall_score(Y_val, val_preds, average='weighted') * 100
    f1 = f1_score(Y_val, val_preds, average='weighted') * 100

    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1
    }

def train_and_evaluate_knn(X_train, Y_train, X_val, Y_val, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, Y_train)

    val_preds = model.predict(X_val)
    accuracy = accuracy_score(Y_val, val_preds) * 100
    recall = recall_score(Y_val, val_preds, average='weighted') * 100
    f1 = f1_score(Y_val, val_preds, average='weighted') * 100

    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1
    }

def k_fold_cross_validation_ml(X, Y, k_folds=5, use_sequential_split=True, model_type='svm', n_neighbors=5):
    X = np.array(X)
    Y = np.array(Y)

    results = []

    if use_sequential_split:
        fold_size = len(X) // k_folds
        indices = list(range(len(X)))

        for fold in range(k_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else len(X)
            val_idx = indices[val_start:val_end]
            train_idx = indices[:val_start] + indices[val_end:]

            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            if model_type == 'svm':
                result = train_and_evaluate_svm(X_train, Y_train, X_val, Y_val)
            elif model_type == 'knn':
                result = train_and_evaluate_knn(X_train, Y_train, X_val, Y_val, n_neighbors=n_neighbors)

            results.append(result)

    else:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            if model_type == 'svm':
                result = train_and_evaluate_svm(X_train, Y_train, X_val, Y_val)
            elif model_type == 'knn':
                result = train_and_evaluate_knn(X_train, Y_train, X_val, Y_val, n_neighbors=n_neighbors)

            results.append(result)

    avg_results = {
        'accuracy': np.mean([res['accuracy'] for res in results]),
        'recall': np.mean([res['recall'] for res in results]),
        'f1_score': np.mean([res['f1_score'] for res in results]),
    }

    print(f"{k_folds}-Fold Cross Validation Results ({model_type.upper()}):")
    print(f"Average Accuracy: {avg_results['accuracy']:.2f}%")
    print(f"Average Recall: {avg_results['recall']:.2f}%")
    print(f"Average F1 Score: {avg_results['f1_score']:.2f}%\n")

    return avg_results

def example_usage():
    # Example Usage
    # Replace these with your actual data
    X_dummy = np.random.rand(100, 10)  # Example feature data
    Y_dummy = np.random.randint(0, 3, size=100)  # Example labels
    
    # SVM Evaluation
    svm_results = k_fold_cross_validation_ml(X_dummy, Y_dummy, k_folds=5, model_type='svm')
    
    # KNN Evaluation
    knn_results = k_fold_cross_validation_ml(X_dummy, Y_dummy, k_folds=5, model_type='knn', n_neighbors=5)
    
    # Save Results to Excel
    results = pd.DataFrame([svm_results, knn_results], index=['SVM', 'KNN'])
    output_path = os.path.join(os.getcwd(), 'Results', 'svm_knn_comparison.xlsx')
    results.to_excel(output_path, index=True, sheet_name='Comparison Results')

def example_usage_():
    import utils
    # ranking and channel selection method
    ranking = utils.get_ranking()
    
    rank_method, channel_selection = 'label_driven_mi', 0.25    
    
    # labels
    y = utils.read_labels(dataset='seed')    
    
    # features
    x = utils.load_cfs_seed('sub1ex1', 'de_LDS', 'joint')
    channels_selected = ranking[rank_method][:int(len(ranking)*channel_selection)]
    
    x_selected = x[:,:,channels_selected]
    x_selected = x_selected.reshape(len(x_selected),-1)
    
    # SVM Evaluation
    svm_results = k_fold_cross_validation_ml(x_selected, y, k_folds=5, model_type='svm')
    
    return svm_results
    
if __name__ == '__main__':
    import drawer_channel_weight
    from utils import utils_feature_loading

    # ranking and channel selection method
    ranking = np.array(drawer_channel_weight.get_index('modeled_g_gaussian'))
    channel_selection = 0.2

    # labels
    y = np.array(utils_feature_loading.read_labels(dataset='seed')).reshape(-1)

    # 获取需要选择的通道
    channels_selected = ranking[:int(len(ranking) * channel_selection)].tolist()

    # 存储结果
    all_results = []

    # 遍历所有 subject (sub1-sub15) 和 experiment (ex1-ex3)
    for sub_id in range(1, 16):
        for ex_id in range(1, 4):
            subject = f'sub{sub_id}'
            experiment = f'ex{ex_id}'
            print(f'Processing: {subject} {experiment}')

            # 读取特征
            x = utils_feature_loading.read_cfs('seed', f'{subject}{experiment}', 'de_LDS', 'joint')
            
            bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']  # or: list(x.keys())
            band_arrays = [x[band] for band in bands]  # 每个是 Point x Channel
            
            # 先堆成 Band x Point x Channel，再转置成 Point x Band x Channel
            x_array = np.stack(band_arrays, axis=0)  # shape: (Band, Point, Channel)
            x_array = np.transpose(x_array, (1, 0, 2))  # shape: (Point, Band, Channel)
            
            # 选择通道
            x_selected = x_array[:, :, channels_selected]
            x_selected = x_selected.reshape(len(x_selected), -1)

            # SVM Evaluation
            svm_results = k_fold_cross_validation_ml(x_selected, y, k_folds=5, model_type='svm')
            all_results.append(svm_results)  # 存储字典

    # 提取所有结果的键
    result_keys = all_results[0].keys()

    # 计算每个指标的平均值
    avg_results = {key: np.mean([res[key] for res in all_results]) for key in result_keys}

    # 输出最终平均结果
    print(f'Average SVM Results: {avg_results}')

    # 保存到 CSV
    df_results = pd.DataFrame(all_results)
    df_results.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in range(1,16) for j in range(1,4)])
    df_results.loc["Average"] = ["Average"] + list(avg_results.values())
    df_results.to_csv("svm_results.csv", index=False)
