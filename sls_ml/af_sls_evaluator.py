import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sls_ml.af_parser import parse_file
from sls_ml.walkaaf import walkaaf, walkaaf_ml, extract_features


# Import your walkaaf implementations



def evaluate_walkaaf(model, X_test, y_test):
    y_pred = model(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def load_test_data():
    test_data_path = '/Users/konraddrees/Documents/GitHub/sls-ml/files/generated_argumentation_frameworks'
    test_files = glob.glob(os.path.join(test_data_path, '*.apx')) + glob.glob(os.path.join(test_data_path, '*.tgf'))

    X_test = []
    y_test = []

    for file_path in test_files:
        af_graph = parse_file(file_path)

        # Extract features and labels
        X = extract_features(af_graph)
        X_test.extend(X)

        stable_labeling = walkaaf(af_graph)
        if stable_labeling:
            y = [stable_labeling[arg] for arg in af_graph.nodes()]
        else:
            # If no stable labeling was found, you can assign some default labels or skip this sample
            # Here we skip the sample
            continue
        y_test.extend(y)

    # Convert the lists to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_test, y_test


if __name__ == '__main__':
    # Load the test dataset
    X_test, y_test = load_test_data()  # Define this function to load your test dataset

    # Evaluate the performance of walkaaf implementations
    walkaaf_results = evaluate_walkaaf(walkaaf, X_test, y_test)
    walkaaf_ml_results = evaluate_walkaaf(walkaaf_ml, X_test, y_test)

    # Print the results
    print("Walkaaf Results:")
    for metric, value in walkaaf_results.items():
        print(f"{metric}: {value}")

    print("\nWalkaaf ML Results:")
    for metric, value in walkaaf_ml_results.items():
        print(f"{metric}: {value}")

    # Visualize the results using bar plots
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    walkaaf_values = [walkaaf_results[metric] for metric in metrics]
    walkaaf_ml_values = [walkaaf_ml_results[metric] for metric in metrics]

    bar_width = 0.35
    index = np.arange(len(metrics))

    plt.bar(index, walkaaf_values, bar_width, label='Walkaaf')
    plt.bar(index + bar_width, walkaaf_ml_values, bar_width, label='Walkaaf ML')

    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Scores')
    plt.title('Performance Comparison of Walkaaf Implementations')
    plt.xticks(index + bar_width / 2, metrics)
    plt.legend()

    plt.show()
