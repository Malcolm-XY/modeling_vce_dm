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

# %% svm foundation
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

# %% example usage
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

# %% example usage
import cw_manager
from utils import utils_feature_loading
def evaluation_cw_control_circle(feature='de_LDS', 
                                 subject_range=range(1, 16), experiment_range=range(1, 4), 
                                 selection_rate=1):
    # manage index of selected channels; features
    channel_weight_df = cw_manager.read_channel_weight_DD('data_driven_pcc', True)
    channel_selected = channel_weight_df.index[:int(len(channel_weight_df.index)*selection_rate)]
    
    # labels
    y = utils_feature_loading.read_labels('seed')
    
    # evaluation circle
    results = []
    for sub in subject_range:
        for ex in experiment_range:
            
            # features
            features = utils_feature_loading.read_cfs('seed', f'sub{sub}ex{ex}', feature)
            alpha = features['alpha'][:, channel_selected]
            beta = features['beta'][:, channel_selected]
            gamma = features['gamma'][:, channel_selected]
            x_selected = np.hstack([alpha, beta, gamma])
            
            # svm evaluation
            svm_results = k_fold_cross_validation_ml(x_selected, y, k_folds=5, model_type='svm')
            
            results.append(svm_results)
    
    print('Evaluation compelete\n')
    
    return results
    
def example_usage_cw_control():
    channel_selection_rate = 0.25
    import cw_manager
    
    channel_weight_df = cw_manager.read_channel_weight_DD('data_driven_pcc', True)
    
    channel_selected = channel_weight_df.index[:int(len(channel_weight_df.index)*channel_selection_rate)]
    
    # labels
    from utils import utils_feature_loading
    y = utils_feature_loading.read_labels('seed')
    
    # features
    features = utils_feature_loading.read_cfs('seed', 'sub3ex1', 'de_LDS')
    alpha = features['alpha'][:, channel_selected]
    beta = features['beta'][:, channel_selected]
    gamma = features['gamma'][:, channel_selected]
    x_selected = np.hstack([alpha, beta, gamma])
    
    # SVM Evaluation
    svm_results = k_fold_cross_validation_ml(x_selected, y, k_folds=5, model_type='svm')
    
    return svm_results

def example_usage_cw_target():
    channel_selection_rate = 0.25
    import cw_manager
    
    channel_weight_df = cw_manager.read_channel_weight_LD('label_driven_mi', True)
    
    channel_selected = channel_weight_df.index[:int(len(channel_weight_df.index)*channel_selection_rate)]
    
    # labels
    from utils import utils_feature_loading
    y = utils_feature_loading.read_labels('seed')
    
    # features
    features = utils_feature_loading.read_cfs('seed', 'sub3ex1', 'de_LDS')
    alpha = features['alpha'][:, channel_selected]
    beta = features['beta'][:, channel_selected]
    gamma = features['gamma'][:, channel_selected]
    x_selected = np.hstack([alpha, beta, gamma])
    
    # SVM Evaluation
    svm_results = k_fold_cross_validation_ml(x_selected, y, k_folds=5, model_type='svm')
    
    return svm_results

def example_usage_cw_fitting():
    channel_selection_rate = 0.25
    import cw_manager
    
    model_fm, model_rcm, model = 'advanced', 'linear', 'powerlaw'
    channel_weight_df = cw_manager.read_channel_weight_fitting(model_fm, model_rcm, model, True)
    
    channel_selected = channel_weight_df.index[:int(len(channel_weight_df.index)*channel_selection_rate)]
    
    # labels
    from utils import utils_feature_loading
    y = utils_feature_loading.read_labels('seed')
    
    # features
    features = utils_feature_loading.read_cfs('seed', 'sub3ex1', 'de_LDS')
    alpha = features['alpha'][:, channel_selected]
    beta = features['beta'][:, channel_selected]
    gamma = features['gamma'][:, channel_selected]
    x_selected = np.hstack([alpha, beta, gamma])
    
    # SVM Evaluation
    svm_results = k_fold_cross_validation_ml(x_selected, y, k_folds=5, model_type='svm')
    
    return svm_results

# %% save

# %%
if __name__ == '__main__':
    # cotrol = example_usage_cw_control()
    # targrt = example_usage_cw_target()
    # fitting = example_usage_cw_fitting()
    
    selection_rate, feature = 0.5, 'de_LDS'
    subject_range, experiment_range = range(11, 16), range(1, 4)
    
    # %% control 1; channel weights computed by the global averaged pcc connectivity matrices from sub1-sub10
    identifier = 'data_driven_pcc_10_15'
    cw_control = cw_manager.read_channel_weight_DD(identifier, sort=True)
    channel_selected_control = cw_control.index[:int(len(cw_control.index)*selection_rate)]    
    
    # labels
    y = utils_feature_loading.read_labels('seed')
    
    # evaluation circle
    results_control = []
    for sub in subject_range:
        for ex in experiment_range:
            
            # features
            features = utils_feature_loading.read_cfs('seed', f'sub{sub}ex{ex}', feature)
            alpha = features['alpha'][:, channel_selected_control]
            beta = features['beta'][:, channel_selected_control]
            gamma = features['gamma'][:, channel_selected_control]
            x_selected = np.hstack([alpha, beta, gamma])
            
            # svm evaluation
            svm_results = k_fold_cross_validation_ml(x_selected, y, k_folds=5, model_type='svm')
            
            results_control.append(svm_results)
    
    print('Evaluation compelete\n')
    
    # calculate average
    result_keys = results_control[0].keys()
    avg_results = {key: np.mean([res[key] for res in results_control]) for key in result_keys}
    print(f'Average SVM Results: {avg_results}')
    
    # save to xlsx
    df_results = pd.DataFrame(results_control)
    df_results.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in subject_range for j in experiment_range])
    df_results.loc["Average"] = ["Average"] + list(avg_results.values())
    
    # 构造保存路径
    path_save = os.path.join(os.getcwd(), 'results_svm_evaluation', 
                             f'svm_results_{feature}_by_{identifier}.xlsx')
    
    # 保存为 Excel 并指定 sheet 名
    with pd.ExcelWriter(path_save, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name=f'selection_rate_{selection_rate}', index=False)
    
    # %% control 2; channel weights computed due to the relevance between channel signals and experiment labels
    identifier = 'label_driven_mi_10_15'
    cw_target = cw_manager.read_channel_weight_LD(identifier, sort=True)
    channel_selected_target = cw_target.index[:int(len(cw_target.index)*selection_rate)]
    
    # labels
    y = utils_feature_loading.read_labels('seed')
    
    # evaluation circle
    results_target = []
    for sub in subject_range:
        for ex in experiment_range:
            
            # features
            features = utils_feature_loading.read_cfs('seed', f'sub{sub}ex{ex}', feature)
            alpha = features['alpha'][:, channel_selected_target]
            beta = features['beta'][:, channel_selected_target]
            gamma = features['gamma'][:, channel_selected_target]
            x_selected = np.hstack([alpha, beta, gamma])
            
            # svm evaluation
            svm_results = k_fold_cross_validation_ml(x_selected, y, k_folds=5, model_type='svm')
            
            results_target.append(svm_results)
    
    print('Evaluation compelete\n')
    
    # calculate average
    result_keys = results_target[0].keys()
    avg_results = {key: np.mean([res[key] for res in results_target]) for key in result_keys}
    print(f'Average SVM Results: {avg_results}')
    
    # save to xlsx
    df_results = pd.DataFrame(results_target)
    df_results.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in subject_range for j in experiment_range])
    df_results.loc["Average"] = ["Average"] + list(avg_results.values())
    
    # 构造保存路径
    path_save = os.path.join(os.getcwd(), 'results_svm_evaluation', 
                             f'svm_results_{feature}_by_{identifier}.xlsx')
    
    # 保存为 Excel 并指定 sheet 名
    with pd.ExcelWriter(path_save, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name=f'selection_rate_{selection_rate}', index=False)
    
    # %% experiment; channel weights computed from the rebuilded connectivity matrix that constructed by vce modeling
    def svm_eval_cw_fitting(model, model_fm, model_rcm, argument='fitting_results(10_15_joint_band_from_mat)', mark='10_15'):
        cw_fitting = cw_manager.read_channel_weight_fitting(model_fm, model_rcm, model, 
                                        source=argument, sort=True)
        channel_selected_fitting = cw_fitting.index[:int(len(cw_fitting.index)*selection_rate)]
        
        # labels
        y = utils_feature_loading.read_labels('seed')
        
        # evaluation circle
        results_fitting = []
        for sub in subject_range:
            for ex in experiment_range:
                
                # features
                features = utils_feature_loading.read_cfs('seed', f'sub{sub}ex{ex}', feature)
                alpha = features['alpha'][:, channel_selected_fitting]
                beta = features['beta'][:, channel_selected_fitting]
                gamma = features['gamma'][:, channel_selected_fitting]
                x_selected = np.hstack([alpha, beta, gamma])
                
                # svm evaluation
                svm_results = k_fold_cross_validation_ml(x_selected, y, k_folds=5, model_type='svm')
                
                results_fitting.append(svm_results)
        
        print('Evaluation compelete\n')
        
        # calculate average
        result_keys = results_fitting[0].keys()
        avg_results = {key: np.mean([res[key] for res in results_fitting]) for key in result_keys}
        print(f'Average SVM Results: {avg_results}')

        # save to xlsx
        # 准备结果数据
        df_results = pd.DataFrame(results_fitting)
        df_results.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in subject_range for j in experiment_range])
        df_results.loc["Average"] = ["Average"] + list(avg_results.values())
        
        # 构造保存路径
        path_save = os.path.join(os.getcwd(), 'results_svm_evaluation', 
                                 f'svm_results_{feature}_by_{model_fm}_{model_rcm}_{mark}.xlsx')
        
        # 判断文件是否存在
        if os.path.exists(path_save):
            # 追加模式，保留已有 sheet，添加新 sheet
            with pd.ExcelWriter(path_save, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_results.to_excel(writer, sheet_name=f'{model}_sr_{selection_rate}', index=False)
        else:
            # 新建文件
            with pd.ExcelWriter(path_save, engine='openpyxl') as writer:
                df_results.to_excel(writer, sheet_name=f'{model}_sr_{selection_rate}', index=False)
                
        return results_fitting
        
    model, model_fm, model_rcm = 'exponential', 'basic', 'differ'
    results = svm_eval_cw_fitting(model, model_fm, model_rcm)
    
    model, model_fm, model_rcm = 'gaussian', 'basic', 'differ'
    results = svm_eval_cw_fitting(model, model_fm, model_rcm)
    
    model, model_fm, model_rcm = 'inverse', 'basic', 'differ'
    results = svm_eval_cw_fitting(model, model_fm, model_rcm)
    
    model, model_fm, model_rcm = 'powerlaw', 'basic', 'differ'
    results = svm_eval_cw_fitting(model, model_fm, model_rcm)
    
    model, model_fm, model_rcm = 'rational_quadratic', 'basic', 'differ'
    results = svm_eval_cw_fitting(model, model_fm, model_rcm)
    
    model, model_fm, model_rcm = 'generalized_gaussian', 'basic', 'differ'
    results = svm_eval_cw_fitting(model, model_fm, model_rcm)
    
    model, model_fm, model_rcm = 'sigmoid', 'basic', 'differ'
    results = svm_eval_cw_fitting(model, model_fm, model_rcm)