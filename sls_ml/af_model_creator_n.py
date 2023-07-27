import os


import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sls_ml.af_parser import parse_file
from collections import Counter


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


def train_models_random(argumentation_folder, processed_feature_folder, processed_label_folder, model_save_folder):
    classifiers = [
        ('DecisionTree', DecisionTreeClassifier()),
        ('NaiveBayes', GaussianNB()),
        ('KNeighbors', KNeighborsClassifier()),
        ('RandomForest', RandomForestClassifier()),
        ('GradientBoosting', GradientBoostingClassifier())
    ]

    argumentation_files = os.listdir(argumentation_folder)
    processed_feature_files = os.listdir(processed_feature_folder)

    # Load the processed feature data from all files
    X_train = pd.DataFrame()  # Initialize an empty dataframe to store the features
    y_train = []

    edge_node_ratios = []  # List to store the edge_node_ratio for each file

    for processed_feature_file in processed_feature_files:
        graph_name = os.path.splitext(processed_feature_file)[0]
        if f'{graph_name}.apx' not in argumentation_files and f'{graph_name}.tgf' not in argumentation_files:
            continue

        # Load the processed feature data from CSV
        processed_feature_file_path = os.path.join(processed_feature_folder, processed_feature_file)
        x_train_data = pd.read_csv(processed_feature_file_path)

        # Load the processed label data from CSV
        processed_label_file_path = os.path.join(processed_label_folder, f'{graph_name}_labels.csv')
        y_train_data = pd.read_csv(processed_label_file_path)

        for arg in x_train_data['argument']:
            feature_row = x_train_data[x_train_data['argument'] == arg].drop('argument', axis=1)
            label_row = y_train_data[y_train_data['Argument'] == arg]

            in_stability_impact = label_row['In_Stability_Impact'].values[0]
            out_stability_impact = label_row['Out_Stability_Impact'].values[0]

            # Create two identical rows for the same argument, one for 'in' and another for 'out'
            feature_row_in = feature_row.copy()
            feature_row_in['state'] = 1  # '1' for 'in' state
            X_train = pd.concat([X_train, feature_row_in], ignore_index=True)

            y_train.append(in_stability_impact)

            feature_row_out = feature_row.copy()
            feature_row_out['state'] = 0  # '0' for 'out' state
            X_train = pd.concat([X_train, feature_row_out], ignore_index=True)

            y_train.append(out_stability_impact)

    X_train1, X_test, y_train1, y_test = train_test_split(X_train, y_train, random_state=42)
    # Train and save models for each classifier
    for classifier_name, classifier in classifiers:
        print(f'{classifier_name} start training')
        model = classifier.fit(X_train1, y_train1)
        print(f'{classifier_name} finished training')
        # Extract feature importances for classifiers that have it available
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            # Make a DataFrame with feature importances and corresponding feature names
            importances_df = pd.DataFrame({
                'Feature': X_train1.columns,
                'Importance': feature_importances
            })
            # Sort by importance
            importances_df = importances_df.sort_values(by='Importance', ascending=False)

            # Save feature importances to a txt file
            with open(f'{classifier_name}_feature_importances.txt', 'w') as f:
                for index, row in importances_df.iterrows():
                    f.write(f"{row['Feature']}: {row['Importance']}\n")
                f.write(f"Total feature importance: {sum(feature_importances)}\n")

        print(f'{classifier_name} start predicting')
        # Predict the labels
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

        # Save the trained model using joblib
        model_file_path = os.path.join(model_save_folder, f'trained_model_{classifier_name}_rn.joblib')
        joblib.dump(model, model_file_path)
        print(f'{classifier_name} saved')
        with open(f'{classifier_name}_metrics.txt', 'w') as f:
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
    processed_label_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_argumentation_frameworks'
    output_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models'

    train_models_random(argumentation_folder, processed_feature_folder, processed_label_folder, output_folder)

