import os
import pandas as pd
import concurrent.futures

from sls_ml.af_parser import parse_file
from sls_ml.af_util import extract_features_graph_arg


def process_graph(file_path, output_folder):
    if file_path.endswith('.tgf') or file_path.endswith('.apx'):
        graph = parse_file(file_path)
        graph_name = os.path.splitext(os.path.basename(file_path))[0]

        # Collect features for all arguments
        all_features = []
        all_arguments = []
        for arg in graph.nodes():
            features = extract_features_graph_arg(graph, arg)
            all_features.append(features)
            all_arguments.append(arg)

        # Create a DataFrame with the collected features and arguments
        data = pd.DataFrame(all_features, columns=['predecessors', 'successors', 'predecessors_neighbors',
                                                   'successors_neighbors', 'degree_centrality',
                                                   'in_degree_centrality', 'out_degree_centrality',
                                                   'closeness_centrality', 'betweenness_centrality',
                                                   'average_neighbor_degree', 'pagerank'])
        data['argument'] = all_arguments

        # Save the DataFrame to a CSV file for the argumentation framework
        output_file = os.path.join(output_folder, f'{graph_name}.csv')
        data.to_csv(output_file, mode='w', header=True, index=False)


def preprocess_data(directory_path, output_folder):
    entries_list = [os.path.join(directory_path, entry) for entry in os.listdir(directory_path)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda entry: process_graph(entry, output_folder), entries_list)

if __name__ == '__main__':
    # paths
    directory_path = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'
    output_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/processed_argumentation_frameworks'

    preprocess_data(directory_path, output_folder)
