import json
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm


def extract_features(file_path):
    data = pd.read_csv(file_path)
    features = data.values
    return features


def extract_labels(file_path):
    data = pd.read_csv(file_path)
    labels = data.values
    return labels


def compute_metrics(X, y, classifier_name):
    return 0


def load_and_preprocess_data_random(processed_feature_file, argumentation_files, processed_feature_folder, processed_label_folder):
    graph_name = os.path.splitext(processed_feature_file)[0]
    if f'{graph_name}.apx' not in argumentation_files and f'{graph_name}.tgf' not in argumentation_files:
        return [], []

    # Load the processed feature data from CSV
    processed_feature_file_path = os.path.join(processed_feature_folder, processed_feature_file)
    x_train_data = pd.read_csv(processed_feature_file_path)

    # Load the processed label data from CSV
    processed_label_file_path = os.path.join(processed_label_folder, f'{graph_name}_labels.csv')
    y_train_data = pd.read_csv(processed_label_file_path)


    x_train_data_dict = x_train_data.set_index('argument').to_dict('index')
    y_train_data_dict = y_train_data.set_index('Argument').to_dict('index')
    X_train = []
    y_train = []

    for arg in x_train_data['argument']:

        feature_row = x_train_data_dict[arg].copy()
        label_row = y_train_data_dict[arg]

        in_stability_impact = label_row['In_Stability_Impact']
        out_stability_impact = label_row['Out_Stability_Impact']

        # Create two identical rows for the same argument, one for 'in' and another for 'out'
        feature_row_in = feature_row.copy()
        feature_row_in['state'] = 1
        X_train.append(feature_row_in)

        y_train.append(in_stability_impact)

        feature_row_out = feature_row.copy()
        feature_row_out['state'] = 0
        X_train.append(feature_row_out)

        y_train.append(out_stability_impact)

    return X_train, y_train


def train_models_random(argumentation_folder, processed_feature_folder, processed_label_folder, model_save_folder):
    classifiers = [
        ('DecisionTree', DecisionTreeClassifier()),
        ('NaiveBayes', GaussianNB()),
        ('KNeighbors', KNeighborsClassifier(n_jobs=-1)),
        ('XGBoost', GridSearchCV(estimator=XGBClassifier(n_jobs=-1),
                                 param_grid={
                                     'learning_rate': [0.01, 0.1, 0.5],
                                     'n_estimators': [100, 200, 500],
                                     'min_child_weight': [1, 5, 10],
                                     'max_depth': [3, 5, 7]},
                                 scoring='roc_auc', cv=5)),
        ('RandomForest', GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                                      param_grid={
                                          'n_estimators': [50, 100, 200],
                                          'max_depth': [None, 10, 20, 30],
                                          'min_samples_leaf': [1, 2, 4],
                                          'min_samples_split': [2, 5, 10]},
                                      scoring='roc_auc', cv=5)),
        ('LightGBM', LGBMClassifier(n_jobs=-1)),
        ('GradientBoosting', GradientBoostingClassifier())
    ]

    argumentation_files = os.listdir(argumentation_folder)
    processed_feature_files = os.listdir(processed_feature_folder)


    X_train = []
    y_train = []


    results = Parallel(n_jobs=-1)(
        delayed(load_and_preprocess_data_random)(processed_feature_file, argumentation_files, processed_feature_folder,
                                                 processed_label_folder)
        for processed_feature_file in tqdm(processed_feature_files)
    )

    for x, y in results:
        X_train.extend(x)
        y_train.extend(y)

    X_train = pd.DataFrame(X_train)
    X_train1, X_test, y_train1, y_test = train_test_split(X_train, y_train, random_state=42)

    # Train and save models for each classifier
    for classifier_name, classifier in classifiers:
        print(f'{classifier_name} start training')
        model = classifier.fit(X_train1, y_train1)
        print(f'{classifier_name} finished training')

        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            importances_df = pd.DataFrame({
                'Feature': X_train1.columns,
                'Importance': feature_importances
            })

            importances_df = importances_df.sort_values(by='Importance', ascending=False)
            # Save feature importances to a txt file
            with open(f'{classifier_name}_rn_feature_importances.txt', 'w') as f:
                for index, row in importances_df.iterrows():
                    f.write(f"{row['Feature']}: {row['Importance']}\n")
                f.write(f"Total feature importance: {sum(feature_importances)}\n")

        print(f'{classifier_name} start predicting')
        y_pred = model.predict(X_test)
        print(f'{classifier_name} finished predicting')

        # Compute the metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan
        report = classification_report(y_test, y_pred, zero_division=0)  # Set zero_division=0
        matrix = confusion_matrix(y_test, y_pred)

        class_balance_ratio = Counter(y_test)
        class_0_count = class_balance_ratio[0]
        class_1_count = class_balance_ratio[1]
        class_balance_ratio = class_0_count / class_1_count if class_1_count != 0 else np.inf
        class_distribution = f"Class 0: {class_0_count} instances, Class 1: {class_1_count} instances"

        # Save the trained model
        model_file_path = os.path.join(model_save_folder, f'trained_model_{classifier_name}_rn.joblib')
        joblib.dump(model, model_file_path)
        print(f'{classifier_name} saved')

        if classifier_name in ['XGBoost', 'RandomForest']:
            best_params = classifier.best_params_
            with open(f'{classifier_name}_rn_best_params.txt', 'w') as f:
                f.write(json.dumps(best_params))

        with open(f'{classifier_name}_rn_metrics.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'ROC AUC: {roc_auc}\n')
            f.write(f'Classification Report:\n{report}\n')
            f.write(f'Confusion Matrix:\n{matrix}\n')
            f.write(f'Class Balance Ratio: {class_balance_ratio}\n')
            f.write(f'Class Distribution: {class_distribution}\n')


def load_and_preprocess_data_in(processed_feature_file, argumentation_files, processed_feature_folder, processed_label_folder):
    X_train = []
    y_train = []

    graph_name = os.path.splitext(processed_feature_file)[0]
    if f'{graph_name}.apx' not in argumentation_files and f'{graph_name}.tgf' not in argumentation_files:
        return [], []

    processed_feature_file_path = os.path.join(processed_feature_folder, processed_feature_file)
    x_train_data = pd.read_csv(processed_feature_file_path)


    processed_label_file_path = os.path.join(processed_label_folder, f'{graph_name}_labels.csv')
    if not os.path.exists(processed_label_file_path):
        return [], []

    y_train_data = pd.read_csv(processed_label_file_path)
    x_train_data_dict = x_train_data.set_index('argument').to_dict('index')
    y_train_data_dict = y_train_data.set_index('Argument').to_dict('index')

    for arg in x_train_data['argument']:
        feature_row = x_train_data_dict[arg].copy()
        label_row = y_train_data_dict[arg]

        label = label_row['Label']
        label = 1 if label == 'in' else 0

        X_train.append(feature_row)
        y_train.append(label)

    return X_train, y_train


def train_models_initial(argumentation_folder, processed_feature_folder, processed_label_folder, model_save_folder):
    classifiers = [
        ('DecisionTree', DecisionTreeClassifier()),
        ('NaiveBayes', GaussianNB()),
        ('KNeighbors', KNeighborsClassifier(n_jobs=-1)),
        ('XGBoost', GridSearchCV(estimator=XGBClassifier(n_jobs=-1),
                                 param_grid={
                                     'learning_rate': [0.01, 0.1, 0.5],
                                     'n_estimators': [100, 200, 500],
                                     'min_child_weight': [1, 5, 10],
                                     'max_depth': [3, 5, 7]},
                                 scoring='roc_auc', cv=5)),
        ('RandomForest', GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                                      param_grid={
                                          'n_estimators': [50, 100, 200],
                                          'max_depth': [None, 10, 20, 30],
                                          'min_samples_leaf': [1, 2, 4],
                                          'min_samples_split': [2, 5, 10]},
                                      scoring='roc_auc', cv=5)),
        ('LightGBM', LGBMClassifier(n_jobs=-1)),
        ('GradientBoosting', GradientBoostingClassifier())
    ]

    argumentation_files = os.listdir(argumentation_folder)
    processed_feature_files = os.listdir(processed_feature_folder)

    X_train = []
    y_train = []

    results = Parallel(n_jobs=-1)(
        delayed(load_and_preprocess_data_in)(processed_feature_file, argumentation_files, processed_feature_folder,
                                          processed_label_folder)
        for processed_feature_file in tqdm(processed_feature_files)
    )

    for x, y in results:
        X_train.extend(x)
        y_train.extend(y)

    X_train = pd.DataFrame(X_train)
    X_train1, X_test, y_train1, y_test = train_test_split(X_train, y_train, random_state=42)
    # Train and save models for each classifier
    for classifier_name, classifier in classifiers:
        print(f'{classifier_name} start training')
        model = classifier.fit(X_train1, y_train1)
        print(f'{classifier_name} finished training')

        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            importances_df = pd.DataFrame({
                'Feature': X_train1.columns,
                'Importance': feature_importances
            })

            importances_df = importances_df.sort_values(by='Importance', ascending=False)
            with open(f'{classifier_name}_in_feature_importances.txt', 'w') as f:
                for index, row in importances_df.iterrows():
                    f.write(f"{row['Feature']}: {row['Importance']}\n")
                f.write(f"Total feature importance: {sum(feature_importances)}\n")

        print(f'{classifier_name} start predicting')
        y_pred = model.predict(X_test)
        print(f'{classifier_name} finished predicting')

        # Compute the metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan
        report = classification_report(y_test, y_pred, zero_division=0)  # Set zero_division=0
        matrix = confusion_matrix(y_test, y_pred)

        class_balance_ratio = Counter(y_test)
        class_0_count = class_balance_ratio[0]
        class_1_count = class_balance_ratio[1]
        class_balance_ratio = class_0_count / class_1_count if class_1_count != 0 else np.inf
        class_distribution = f"Class 0: {class_0_count} instances, Class 1: {class_1_count} instances"

        # Save the trained model
        model_file_path = os.path.join(model_save_folder, f'trained_model_{classifier_name}_in.joblib')
        joblib.dump(model, model_file_path)
        print(f'{classifier_name} saved')

        if classifier_name in ['XGBoost', 'RandomForest']:
            best_params = classifier.best_params_
            with open(f'{classifier_name}_in_best_params.txt', 'w') as f:
                f.write(json.dumps(best_params))

        with open(f'{classifier_name}_in_metrics.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'ROC AUC: {roc_auc}\n')
            f.write(f'Classification Report:\n{report}\n')
            f.write(f'Confusion Matrix:\n{matrix}\n')
            f.write(f'Class Balance Ratio: {class_balance_ratio}\n')
            f.write(f'Class Distribution: {class_distribution}\n')


def train_models_initial_reduced(argumentation_folder, processed_feature_folder, processed_label_folder, model_save_folder):
    classifiers = [
        ('XGBoost', GridSearchCV(estimator=XGBClassifier(n_jobs=-1),
                                 param_grid={
                                     'learning_rate': [0.01, 0.1, 0.5],
                                     'n_estimators': [100, 200, 500],
                                     'min_child_weight': [1, 5, 10],
                                     'max_depth': [3, 5, 7]},
                                 scoring='roc_auc', cv=5)),
        ('RandomForest', GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                                      param_grid={
                                          'n_estimators': [50, 100, 200],
                                          'max_depth': [None, 10, 20, 30],
                                          'min_samples_leaf': [1, 2, 4],
                                          'min_samples_split': [2, 5, 10]},
                                      scoring='roc_auc', cv=5))
    ]

    argumentation_files = os.listdir(argumentation_folder)
    processed_feature_files = os.listdir(processed_feature_folder)

    X_train = []
    y_train = []

    results = Parallel(n_jobs=-1)(
        delayed(load_and_preprocess_data_in)(processed_feature_file, argumentation_files, processed_feature_folder,
                                          processed_label_folder)
        for processed_feature_file in tqdm(processed_feature_files)
    )

    for x, y in results:
        X_train.extend(x)
        y_train.extend(y)

    X_train = pd.DataFrame(X_train)
    X_train.drop(['in_degree_centrality', 'predecessors_neighbors', 'degree_centrality','average_neighbor_degree','out_degree_centrality', 'successors_neighbors', 'edge_node_ratio', 'successors' ], axis=1, inplace=True)

    X_train1, X_test, y_train1, y_test = train_test_split(X_train, y_train, random_state=42)
    # Train and save models for each classifier
    for classifier_name, classifier in classifiers:
        print(f'{classifier_name} start training')
        model = classifier.fit(X_train1, y_train1)
        print(f'{classifier_name} finished training')

        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            importances_df = pd.DataFrame({
                'Feature': X_train1.columns,
                'Importance': feature_importances
            })

            importances_df = importances_df.sort_values(by='Importance', ascending=False)

            with open(f'{classifier_name}_in_feature_importances.txt', 'w') as f:
                for index, row in importances_df.iterrows():
                    f.write(f"{row['Feature']}: {row['Importance']}\n")
                f.write(f"Total feature importance: {sum(feature_importances)}\n")

        print(f'{classifier_name} start predicting')

        y_pred = model.predict(X_test)
        print(f'{classifier_name} finished predicting')

        # Compute the metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan
        report = classification_report(y_test, y_pred, zero_division=0)  # Set zero_division=0
        matrix = confusion_matrix(y_test, y_pred)

        class_balance_ratio = Counter(y_test)
        class_0_count = class_balance_ratio[0]
        class_1_count = class_balance_ratio[1]
        class_balance_ratio = class_0_count / class_1_count if class_1_count != 0 else np.inf
        class_distribution = f"Class 0: {class_0_count} instances, Class 1: {class_1_count} instances"

        # Save the trained model
        model_file_path = os.path.join(model_save_folder, f'trained_model_{classifier_name}_in_red.joblib')
        joblib.dump(model, model_file_path)
        print(f'{classifier_name} saved')

        if classifier_name in ['XGBoost', 'RandomForest']:
            best_params = classifier.best_params_
            with open(f'{classifier_name}_in_best_params_red.txt', 'w') as f:
                f.write(json.dumps(best_params))

        with open(f'{classifier_name}_in_metrics_red.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'ROC AUC: {roc_auc}\n')
            f.write(f'Classification Report:\n{report}\n')
            f.write(f'Confusion Matrix:\n{matrix}\n')
            f.write(f'Class Balance Ratio: {class_balance_ratio}\n')
            f.write(f'Class Distribution: {class_distribution}\n')


def train_models_random_reduced(argumentation_folder, processed_feature_folder, processed_label_folder, model_save_folder):
    classifiers = [

        ('XGBoost', GridSearchCV(estimator=XGBClassifier(n_jobs=-1),
                                 param_grid={
                                     'learning_rate': [0.01, 0.1, 0.5],
                                     'n_estimators': [100, 200, 500],
                                     'min_child_weight': [1, 5, 10],
                                     'max_depth': [3, 5, 7]},
                                 scoring='roc_auc', cv=5)),
        ('RandomForest', GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                                      param_grid={
                                          'n_estimators': [50, 100, 200],
                                          'max_depth': [None, 10, 20, 30],
                                          'min_samples_leaf': [1, 2, 4],
                                          'min_samples_split': [2, 5, 10]},
                                      scoring='roc_auc', cv=5))
    ]

    argumentation_files = os.listdir(argumentation_folder)
    processed_feature_files = os.listdir(processed_feature_folder)

    X_train = []
    y_train = []

    results = Parallel(n_jobs=-1)(
        delayed(load_and_preprocess_data_random)(processed_feature_file, argumentation_files, processed_feature_folder,
                                                 processed_label_folder)
        for processed_feature_file in tqdm(processed_feature_files)  # Add tqdm progress bar here
    )

    for x, y in results:
        X_train.extend(x)
        y_train.extend(y)

    X_train = pd.DataFrame(X_train)
    X_train.drop(['in_degree_centrality', 'predecessors_neighbors', 'degree_centrality','average_neighbor_degree','out_degree_centrality', 'successors_neighbors', 'edge_node_ratio', 'successors' ], axis=1, inplace=True)


    X_train1, X_test, y_train1, y_test = train_test_split(X_train, y_train, random_state=42)

    for classifier_name, classifier in classifiers:
        print(f'{classifier_name} start training')
        model = classifier.fit(X_train1, y_train1)
        print(f'{classifier_name} finished training')
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_

            importances_df = pd.DataFrame({
                'Feature': X_train1.columns,
                'Importance': feature_importances
            })

            importances_df = importances_df.sort_values(by='Importance', ascending=False)
            with open(f'{classifier_name}_rn_feature_importances.txt', 'w') as f:
                for index, row in importances_df.iterrows():
                    f.write(f"{row['Feature']}: {row['Importance']}\n")
                f.write(f"Total feature importance: {sum(feature_importances)}\n")

        print(f'{classifier_name} start predicting')
        y_pred = model.predict(X_test)
        print(f'{classifier_name} finished predicting')
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan
        report = classification_report(y_test, y_pred, zero_division=0)  # Set zero_division=0
        matrix = confusion_matrix(y_test, y_pred)

        class_balance_ratio = Counter(y_test)
        class_0_count = class_balance_ratio[0]
        class_1_count = class_balance_ratio[1]
        class_balance_ratio = class_0_count / class_1_count if class_1_count != 0 else np.inf
        class_distribution = f"Class 0: {class_0_count} instances, Class 1: {class_1_count} instances"

        # Save the trained
        model_file_path = os.path.join(model_save_folder, f'trained_model_{classifier_name}_rn_red.joblib')
        joblib.dump(model, model_file_path)
        print(f'{classifier_name} saved')

        if classifier_name in ['XGBoost', 'RandomForest']:
            best_params = classifier.best_params_
            with open(f'{classifier_name}_rn_best_params_red.txt', 'w') as f:
                f.write(json.dumps(best_params))

        with open(f'{classifier_name}_rn_metrics_red.txt', 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'ROC AUC: {roc_auc}\n')
            f.write(f'Classification Report:\n{report}\n')
            f.write(f'Confusion Matrix:\n{matrix}\n')
            f.write(f'Class Balance Ratio: {class_balance_ratio}\n')
            f.write(f'Class Distribution: {class_distribution}\n')


if __name__ == '__main__':
    # Paths
    argumentation_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'
    processed_feature_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/processed_argumentation_frameworks'

    processed_label_rn_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_argumentation_frameworks'
    processed_label_in_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_initial_argumentation_frameworks'

    output_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models'

    train_models_random(argumentation_folder, processed_feature_folder, processed_label_rn_folder, output_folder)
    train_models_initial(argumentation_folder,processed_feature_folder,processed_label_in_folder,output_folder)
    train_models_random_reduced(argumentation_folder, processed_feature_folder, processed_label_rn_folder, output_folder)
    train_models_initial_reduced(argumentation_folder,processed_feature_folder,processed_label_in_folder,output_folder)