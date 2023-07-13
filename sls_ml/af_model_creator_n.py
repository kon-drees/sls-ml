import os


import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sls_ml.af_parser import parse_file



#node edge ratio
#length out in? max?
# vol


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



def train_models(argumentation_folder, processed_feature_folder, processed_label_folder, model_save_folder):
    classifiers = [
        ('DecisionTree', DecisionTreeClassifier()),
        ('NaiveBayes', GaussianNB()),
        ('KNeighbors', KNeighborsClassifier()),
        ('RandomForest', RandomForestClassifier())
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

        # Calculate edge_node_ratio for this file
        graph_file = os.path.join(argumentation_folder, f'{graph_name}.apx')
        if not os.path.exists(graph_file):
            graph_file = os.path.join(argumentation_folder, f'{graph_name}.tgf')

        argumentation_framework = parse_file(graph_file)
        edge_node_ratio = argumentation_framework.number_of_edges() / argumentation_framework.number_of_nodes()
        edge_node_ratios.append(edge_node_ratio)

        for arg in x_train_data['argument']:
            feature_row = x_train_data[x_train_data['argument'] == arg].drop('argument', axis=1)
            label_row = y_train_data[y_train_data['Argument'] == arg]

            in_stability_impact = label_row['In_Stability_Impact'].values[0]
            out_stability_impact = label_row['Out_Stability_Impact'].values[0]

            feature_row['edge_node_ratio'] = edge_node_ratio  # Add edge_node_ratio to the feature row

            # Create two identical rows for the same argument, one for 'in' and another for 'out'
            feature_row_in = feature_row.copy()
            feature_row_in['state'] = 1  # '1' for 'in' state
            X_train = pd.concat([X_train, feature_row_in], ignore_index=True)

            y_train.append(in_stability_impact)

            feature_row_out = feature_row.copy()
            feature_row_out['state'] = 0  # '0' for 'out' state
            X_train = pd.concat([X_train, feature_row_out], ignore_index=True)

            y_train.append(out_stability_impact)

    # Train and save models for each classifier
    for classifier_name, classifier in classifiers:
        model = classifier.fit(X_train, y_train)

        # Save the trained model using joblib
        model_file_path = os.path.join(model_save_folder, f'trained_model_{classifier_name}.joblib')
        joblib.dump(model, model_file_path)


if __name__ == '__main__':
    # Paths
    argumentation_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'
    processed_feature_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/processed_argumentation_frameworks'
    processed_label_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_argumentation_frameworks'
    output_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models'

    train_models(argumentation_folder, processed_feature_folder, processed_label_folder, output_folder)
