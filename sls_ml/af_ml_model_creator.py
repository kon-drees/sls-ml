import os
import concurrent.futures
import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


from sls_ml.af_parser import parse_file
from sls_ml.walkaaf import get_random_labeling, get_mislabeled_args




def extract_features(graph, arg):
    features = [
        len(list(graph.predecessors(arg))),
        len(list(graph.successors(arg))),
        sum(len(list(graph.predecessors(neighbor))) for neighbor in graph.predecessors(arg)),
        sum(len(list(graph.successors(neighbor))) for neighbor in graph.successors(arg)),
        nx.degree_centrality(graph)[arg],
        nx.in_degree_centrality(graph)[arg],
        nx.out_degree_centrality(graph)[arg],
        nx.closeness_centrality(graph)[arg],
        nx.betweenness_centrality(graph)[arg],
        nx.average_neighbor_degree(graph)[arg],
        nx.pagerank(graph)[arg],
        nx.hits(graph)[arg],
        nx.eigenvector_centrality(graph)[arg]
    ]
    return features


def create_labels(graph):
    labels = {}

    for arg in tqdm(graph.nodes(), desc='Creating labels', position=0, leave=False):
        labeling = get_random_labeling(set(graph.nodes))
        flipped_labeling = labeling.copy()
        flipped_labeling[arg] = 'out' if flipped_labeling[arg] == 'in' else 'in'

        mislabeled_before_flip = len(get_mislabeled_args(labeling, graph))
        mislabeled_after_flip = len(get_mislabeled_args(flipped_labeling, graph))

        # Stability Impact
        stability_impact = mislabeled_after_flip - mislabeled_before_flip

        # Influence Impact
        influence_before_flip = len(list(graph.successors(arg)))
        influence_after_flip = len(list(graph.successors(arg))) if flipped_labeling[arg] == 'in' else 0
        influence_impact = influence_after_flip - influence_before_flip

        # Conflict Impact
        conflict_before_flip = len(list(graph.predecessors(arg)))
        conflict_after_flip = len(list(graph.predecessors(arg))) if flipped_labeling[arg] == 'in' else 0
        conflict_impact = conflict_after_flip - conflict_before_flip

        labels[arg] = (stability_impact, influence_impact, conflict_impact)

    return labels


def process_graph(file_path, directory_path, output_file='all_features_labels.csv'):
    X = []
    y = []

    if file_path.endswith('.tgf') or file_path.endswith('.apx'):
        graph = parse_file(file_path)
        labels = create_labels(graph)
        for arg in graph.nodes():
            features = extract_features(graph, arg)
            X.append(features)
            y.append(labels[arg])

        # Save the features and labels in a single CSV file
        file_name = os.path.basename(file_path)

        data = pd.DataFrame(X, columns=['predecessors', 'successors', 'predecessors_neighbors', 'successors_neighbors',
                                        'degree_centrality', 'in_degree_centrality', 'out_degree_centrality',
                                        'closeness_centrality', 'betweenness_centrality', 'average_neighbor_degree',
                                        'pagerank', 'hits', 'eigenvector_centrality'])
        data['stability_impact'] = [label[0] for label in y]
        data['influence_impact'] = [label[1] for label in y]
        data['conflict_impact'] = [label[2] for label in y]
        data['graph_name'] = file_name

        # Append to the existing CSV file or create it if it doesn't exist
        if os.path.isfile(output_file):
            data.to_csv(output_file, mode='a', header=False, index=False)
        else:
            data.to_csv(output_file, mode='w', header=True, index=False)

    return X, y


def preprocess_data(directory_path):
    X = []
    y = []

    entries_list = [os.path.join(directory_path, entry) for entry in os.listdir(directory_path)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_graph, entries_list, [directory_path]*len(entries_list)), total=len(entries_list), desc='Preprocessing data', position=0, leave=False))

    for result in results:
        X.extend(result[0])
        y.extend(result[1])

    return X, y


def train_and_save_model(X, y, model_path='ml_model.joblib', metrics_path='metrics.txt', selected_features_path='selected_features.txt'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = [
        ('DecisionTree', DecisionTreeClassifier()),
        ('NaiveBayes', GaussianNB()),
        ('KNeighbors', KNeighborsClassifier()),
        ('RandomForest', RandomForestClassifier())
    ]

    for classifier_name, classifier in classifiers:
        # Create the RFE selector and fit it to the training data
        selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=4)
        selector.fit(X_train, y_train)

        # Save the selected features and their indices
        with open(f'{classifier_name}_{selected_features_path}', 'w') as f:
            f.write("Selected features indices:\n")
            f.write(str(selector.get_support(indices=True)) + '\n')


        # Transform both the training and test datasets to contain only the selected features
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)

        # Train the model using the transformed training dataset
        classifier.fit(X_train_selected, y_train)

        # Predict the labels using the transformed test dataset
        y_pred = classifier.predict(X_test_selected)
        print(f'{classifier_name} Accuracy:', accuracy_score(y_test, y_pred))

        # Compute and print the additional metrics
        print(f"{classifier_name} Classification report:")
        report = classification_report(y_test, y_pred)
        print(report)
        roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
        print(f"{classifier_name} ROC AUC score:", roc_auc)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"{classifier_name} Confusion matrix:")
        print(conf_matrix)

        # Save the metrics to disk
        with open(f'{classifier_name}_{metrics_path}', 'w') as f:
            f.write(f'Accuracy: {accuracy_score(y_test, y_pred)}\n')
            f.write(f'Classification report:\n{report}\n')
            f.write(f'ROC AUC score: {roc_auc}\n')
            f.write(f'Confusion matrix:\n{conf_matrix}\n')

        # Save the model to disk
        joblib.dump(classifier, f'{classifier_name}_{model_path}')


def load_features_labels(input_file='features_labels.csv'):
    data = pd.read_csv(input_file)
    X = data.drop(['stability_impact', 'influence_impact', 'conflict_impact'], axis=1).values.tolist()
    y = data[['stability_impact', 'influence_impact', 'conflict_impact']].values.tolist()

    return X, y



if __name__ == '__main__':
    directory_path = '/Users/konraddrees/Documents/GitHub/sls-ml/files/training_data'

    # Uncomment the line below if you want to preprocess the data and save it to a CSV file
    # X, y =
    preprocess_data(directory_path)

    # Load the features and labels from the CSV file
    # X, y = load_features_labels()

    #  train_and_save_model(X, y)


