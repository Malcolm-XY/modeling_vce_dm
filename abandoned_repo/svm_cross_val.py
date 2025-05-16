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
    import feature_engineering
    from utils import utils_feature_loading
    from utils import utils_visualization
    # labels
    y = utils_feature_loading.read_labels(dataset='seed')    
    
    # VCE model
    import vce_modeling
    _, distance_matrix = feature_engineering.compute_distance_matrix('seed',
        'euclidean', stereo_params={'prominence': 0.5, 'epsilon': 0.01}, visualize=True)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)
    utils_visualization.draw_projection(distance_matrix)
    
    factor_matrix = vce_modeling.compute_volume_conduction_factors(distance_matrix, 
                                                                   'sigmoid', {'mu': 2.0, 'beta': 1.0})
    factor_matrix_normalized = feature_engineering.normalize_matrix(factor_matrix)
    utils_visualization.draw_projection(factor_matrix_normalized)
    
    # features
    features = utils_feature_loading.read_fcs_mat('seed', 'sub1ex1', 'pcc')
    alpha = features['alpha']
    beta = features['beta']
    gamma = features['gamma']
    
    tril_indices = np.tril_indices(62)
    alpha_lower = alpha[:, tril_indices[0], tril_indices[1]]
    beta_lower = beta[:, tril_indices[0], tril_indices[1]]
    gamma_lower = gamma[:, tril_indices[0], tril_indices[1]]
    
    x = np.hstack((alpha_lower, beta_lower, gamma_lower))
    
    # recovered features
    alpha_recovered = alpha - factor_matrix_normalized
    beta_recovered = beta - factor_matrix_normalized
    gamma_recovered = gamma - factor_matrix_normalized
    
    tril_indices = np.tril_indices(62)
    alpha_recovered_lower = alpha_recovered[:, tril_indices[0], tril_indices[1]]
    beta_recovered_lower = beta_recovered[:, tril_indices[0], tril_indices[1]]
    gamma_recovered_lower = gamma_recovered[:, tril_indices[0], tril_indices[1]]
    
    x_recovered = np.hstack((alpha_recovered_lower, beta_recovered_lower, gamma_recovered_lower))
    
    # SVM Evaluation
    svm_results = k_fold_cross_validation_ml(x, y, k_folds=5, model_type='svm')
    svm_results_ = k_fold_cross_validation_ml(x_recovered, y, k_folds=5, model_type='svm')
    
    return svm_results, svm_results_
    
def valid_all_subjects():
    import feature_engineering
    import vce_modeling
    from utils import utils_feature_loading, utils_visualization
    
    # labels
    y = utils_feature_loading.read_labels(dataset='seed')
    
    # VCE model: distance matrix & factor matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix(
        'seed', 'euclidean', stereo_params={'prominence': 0.5, 'epsilon': 0.01}, visualize=True)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)
    utils_visualization.draw_projection(distance_matrix)
    
    factor_matrix = vce_modeling.compute_volume_conduction_factors(
        distance_matrix, 'sigmoid', {'mu': 2.0, 'beta': 1.0})
    factor_matrix_normalized = feature_engineering.normalize_matrix(factor_matrix)
    utils_visualization.draw_projection(factor_matrix_normalized)

    tril_indices = np.tril_indices(62)

    all_results_original = []
    all_results_recovered = []

    for sub in range(1, 16):
        for ex in range(1, 4):
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            features = utils_feature_loading.read_fcs_mat('seed', subject_id, 'pcc')
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # Original features
            alpha_lower = alpha[:, tril_indices[0], tril_indices[1]]
            beta_lower = beta[:, tril_indices[0], tril_indices[1]]
            gamma_lower = gamma[:, tril_indices[0], tril_indices[1]]
            x = np.hstack((alpha_lower, beta_lower, gamma_lower))

            # Recovered features
            alpha_recovered = alpha - factor_matrix_normalized
            beta_recovered = beta - factor_matrix_normalized
            gamma_recovered = gamma - factor_matrix_normalized
            alpha_recovered_lower = alpha_recovered[:, tril_indices[0], tril_indices[1]]
            beta_recovered_lower = beta_recovered[:, tril_indices[0], tril_indices[1]]
            gamma_recovered_lower = gamma_recovered[:, tril_indices[0], tril_indices[1]]
            x_recovered = np.hstack((alpha_recovered_lower, beta_recovered_lower, gamma_recovered_lower))

            # SVM Evaluation
            svm_results = k_fold_cross_validation_ml(x, y, k_folds=5, model_type='svm')
            svm_results_recovered = k_fold_cross_validation_ml(x_recovered, y, k_folds=5, model_type='svm')

            all_results_original.append(svm_results)           
            all_results_recovered.append(svm_results_recovered)
        
    # Avg results
    result_original_keys = all_results_original[0].keys()
    avg_results_original = {key: np.mean([res[key] for res in all_results_original]) for key in result_original_keys}
    print(f'Average SVM Results (CM): {avg_results_original}')
    
    result_recovered_keys = all_results_recovered[0].keys()
    avg_results_recovered = {key: np.mean([res[key] for res in all_results_recovered]) for key in result_recovered_keys}
    print(f'Average SVM Results (Recovered CM): {avg_results_recovered}')
    
    # Save to CSV
    df_results_original = pd.DataFrame(all_results_original)
    df_results_original.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in range(1,16) for j in range(1,4)])
    df_results_original.loc["Average"] = ["Average"] + list(avg_results_original.values())
    df_results_original.to_csv("svm_results_CM.csv", index=False)
    
    df_results_recovered = pd.DataFrame(all_results_recovered)
    df_results_recovered.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in range(1,16) for j in range(1,4)])
    df_results_recovered.loc["Average"] = ["Average"] + list(avg_results_recovered.values())
    df_results_recovered.to_csv("svm_results_RCM.csv", index=False)
        
    return all_results_original, all_results_recovered

if __name__ == '__main__':
    import feature_engineering
    import vce_modeling
    from utils import utils_feature_loading, utils_visualization
    
    # labels
    y = utils_feature_loading.read_labels(dataset='seed')
    
    # VCE model: distance matrix & factor matrix
    _, distance_matrix = feature_engineering.compute_distance_matrix(
        'seed', 'euclidean', stereo_params={'prominence': 0.5, 'epsilon': 0.01}, visualize=True)
    distance_matrix = feature_engineering.normalize_matrix(distance_matrix)
    utils_visualization.draw_projection(distance_matrix)
    
    factor_matrix = vce_modeling.compute_volume_conduction_factors(
        distance_matrix, 'sigmoid', {'mu': 2.0, 'beta': 1.0})
    factor_matrix_normalized = feature_engineering.normalize_matrix(factor_matrix)
    utils_visualization.draw_projection(factor_matrix_normalized)

    tril_indices = np.tril_indices(62)

    all_results_original = []
    all_results_recovered = []

    for sub in range(1, 16):
        for ex in range(1, 4):
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            features = utils_feature_loading.read_fcs_mat('seed', subject_id, 'plv')
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # Original features
            alpha_lower = alpha[:, tril_indices[0], tril_indices[1]]
            beta_lower = beta[:, tril_indices[0], tril_indices[1]]
            gamma_lower = gamma[:, tril_indices[0], tril_indices[1]]
            x = np.hstack((alpha_lower, beta_lower, gamma_lower))

            # Recovered features
            alpha_recovered = alpha - factor_matrix_normalized
            beta_recovered = beta - factor_matrix_normalized
            gamma_recovered = gamma - factor_matrix_normalized
            alpha_recovered_lower = alpha_recovered[:, tril_indices[0], tril_indices[1]]
            beta_recovered_lower = beta_recovered[:, tril_indices[0], tril_indices[1]]
            gamma_recovered_lower = gamma_recovered[:, tril_indices[0], tril_indices[1]]
            x_recovered = np.hstack((alpha_recovered_lower, beta_recovered_lower, gamma_recovered_lower))

            # SVM Evaluation
            svm_results = k_fold_cross_validation_ml(x, y, k_folds=5, model_type='svm')
            svm_results_recovered = k_fold_cross_validation_ml(x_recovered, y, k_folds=5, model_type='svm')

            all_results_original.append(svm_results)           
            all_results_recovered.append(svm_results_recovered)
        
    # Avg results
    result_original_keys = all_results_original[0].keys()
    avg_results_original = {key: np.mean([res[key] for res in all_results_original]) for key in result_original_keys}
    print(f'Average SVM Results (CM): {avg_results_original}')
    
    result_recovered_keys = all_results_recovered[0].keys()
    avg_results_recovered = {key: np.mean([res[key] for res in all_results_recovered]) for key in result_recovered_keys}
    print(f'Average SVM Results (Recovered CM): {avg_results_recovered}')
    
    # Save to CSV
    df_results_original = pd.DataFrame(all_results_original)
    df_results_original.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in range(1,16) for j in range(1,4)])
    df_results_original.loc["Average"] = ["Average"] + list(avg_results_original.values())
    df_results_original.to_csv("svm_results_CM.csv", index=False)
    
    df_results_recovered = pd.DataFrame(all_results_recovered)
    df_results_recovered.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in range(1,16) for j in range(1,4)])
    df_results_recovered.loc["Average"] = ["Average"] + list(avg_results_recovered.values())
    df_results_recovered.to_csv("svm_results_RCM.csv", index=False)
