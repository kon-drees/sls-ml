import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sls_ml.af_parser import parse_file
from sls_ml.af_util import get_mislabeled_args, get_random_labeling
from joblib import dump







def extract_features(file_path):
    data = pd.read_csv(file_path)
    features = data.values[:, 1:]  # Exclude the 'argument' column
    return features







def train_and_save_model(X, y, classifier_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if classifier_name == 'DecisionTree':
        classifier = DecisionTreeClassifier()
    elif classifier_name == 'RandomForest':
        classifier = RandomForestClassifier()
    elif classifier_name == 'KNeighbors':
        classifier = KNeighborsClassifier()
    elif classifier_name == 'NaiveBayes':
        classifier = GaussianNB()

    # Train the model
    classifier.fit(X_train, y_train)

    # Predict the labels
    y_pred = classifier.predict(X_test)

    # Compute the metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan
    report = classification_report(y_test, y_pred, zero_division=0)  # Set zero_division=0
    matrix = confusion_matrix(y_test, y_pred)

    # Save the model
    dump(classifier, f'{classifier_name}_model.joblib')

    # Save the metrics
    with open(f'{classifier_name}_metrics.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'ROC AUC: {roc_auc}\n')
        f.write(f'Classification Report:\n{report}\n')
        f.write(f'Confusion Matrix:\n{matrix}\n')


def create_labels(graph):

    labels = {}
    ml_labeling = {}

    for _ in range(50):
        initial_labeling = get_random_labeling(set(graph.nodes()))

        mislabeled_args = get_mislabeled_args(initial_labeling, graph)

        for arg in mislabeled_args:
            labeling = initial_labeling.copy()
            flipped_labeling = labeling.copy()
            flipped_labeling[arg] = 'out' if flipped_labeling[arg] == 'in' else 'in'

            mislabeled_before_flip = len(get_mislabeled_args(labeling, graph))
            mislabeled_after_flip = len(get_mislabeled_args(flipped_labeling, graph))

            stability_impact = mislabeled_before_flip - mislabeled_after_flip

            if arg in labels:
                labels[arg].append(stability_impact)
            else:
                labels[arg] = [stability_impact]

    initial_labeling = get_random_labeling(set(graph.nodes()))

    # Making sure that every node has at least one value
    for arg in graph.nodes():
        labeling = initial_labeling.copy()
        flipped_labeling = labeling.copy()
        flipped_labeling[arg] = 'out' if flipped_labeling[arg] == 'in' else 'in'

        mislabeled_before_flip = len(get_mislabeled_args(labeling, graph))
        mislabeled_after_flip = len(get_mislabeled_args(flipped_labeling, graph))

        stability_impact = mislabeled_before_flip - mislabeled_after_flip

        if arg in labels:
            labels[arg].append(stability_impact)
        else:
            labels[arg] = [stability_impact]

    for arg in labels:
        stability_impacts = labels[arg]

        # Minimum Stability Impact
        # min_stability = min(stability_impacts)
        # ml_labeling[arg] = [min_stability]

        # Maximum Stability Impact
        # max_stability = max(stability_impacts)
        # ml_labeling[arg].append(max_stability)

        # Count positive and negative stability impacts
        positive_impacts = sum(1 for impact in stability_impacts if impact > 0)
        negative_impacts = sum(1 for impact in stability_impacts if impact <= 0)

        # Majority impact
        majority_vote = 1 if positive_impacts > negative_impacts else 0
        ml_labeling[arg].append(majority_vote)

        # Average Stability Impact
        # avg_stability = sum(stability_impacts) / len(stability_impacts)
        # ml_labeling[arg].append(avg_stability)

    return ml_labeling


if __name__ == '__main__':
    # Paths
    argumentation_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'
    processed_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/processed_argumentation_frameworks'
    output_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models'

    # Read the argumentation frameworks and create labels
    argumentation_files = os.listdir(argumentation_folder)
    processed_files = os.listdir(processed_folder)

    # Initialize empty lists for features and labels
    X_all = []
    y_all = []

    num_labelings_per_framework = 10  # Adjust the number of labelings per framework as desired

    for processed_file in processed_files:
        graph_name = os.path.splitext(processed_file)[0]
        if f'{graph_name}.apx' not in argumentation_files and f'{graph_name}.tgf' not in argumentation_files:
            continue

        # Load the processed data and extract features
        processed_data = extract_features(os.path.join(processed_folder, processed_file))

        graph_file = os.path.join(argumentation_folder, f'{graph_name}.apx')
        if not os.path.exists(graph_file):
            graph_file = os.path.join(argumentation_folder, f'{graph_name}.tgf')

        argumentation_framework = parse_file(graph_file)
        labels = create_labels(argumentation_framework)

        # Append features and labels to the lists
        print(graph_name)
        print(labels)

        X_all.append(processed_data)
        y_all.append([1 if np.mean(labels[arg]) >= 0 else 0 for arg in argumentation_framework.nodes()])


    # Concatenate the features and labels
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    print(y)
    classifiers = [
        ('DecisionTree', DecisionTreeClassifier()),
        ('NaiveBayes', GaussianNB()),
        ('KNeighbors', KNeighborsClassifier()),
        ('RandomForest', RandomForestClassifier())
    ]

    for classifier_name, classifier in classifiers:
        # Train and save the model
        train_and_save_model(X, y, classifier_name)