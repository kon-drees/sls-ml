import os
import pandas as pd
import concurrent.futures
import signal

from tqdm import tqdm
from sls_ml.af_parser import parse_file
from sls_ml.af_util import extract_features_graph_arg


def process_graph(file_path, output_folder, processed_files, pbar):
    if file_path.endswith('.tgf') or file_path.endswith('.apx'):
        graph = parse_file(file_path)
        graph_name = os.path.splitext(os.path.basename(file_path))[0]

        if graph_name in processed_files:
            return


        features = []
        arguments = []
        total_arguments = len(graph.nodes())

        with tqdm(total=total_arguments, desc=f'Framework: {graph_name}', leave=False) as pbar:
            for arg in graph.nodes():
                arguments.append(arg)
                arg_features = extract_features_graph_arg(graph, arg)
                features.append(arg_features)

                # Update the progress bar for the current argument
                pbar.set_postfix({'Argument': arg})
                pbar.update(1)

        # Create a DataFrame with the collected features
        data = pd.DataFrame(features, columns=['predecessors', 'successors', 'predecessors_neighbors',
                                               'successors_neighbors', 'degree_centrality',
                                               'in_degree_centrality', 'out_degree_centrality',
                                               'closeness_centrality', 'betweenness_centrality',
                                               'average_neighbor_degree', 'pagerank'])
        data.insert(0, 'argument', arguments)

        output_file = os.path.join(output_folder, f'{graph_name}.csv')
        data.to_csv(output_file, mode='w', header=True, index=False)


        processed_files.add(graph_name)


        pbar.update(1)


def preprocess_data(directory_path, output_folder, processed_files_file):
    processed_files = set()

    if os.path.exists(processed_files_file):
        with open(processed_files_file, 'r') as file:
            processed_files = set(file.read().splitlines())

    entries_list = [os.path.join(directory_path, entry) for entry in os.listdir(directory_path)]

    with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(total=len(entries_list), desc='Progress') as pbar:
        futures = [executor.submit(process_graph, entry, output_folder, processed_files, pbar) for entry in entries_list]

        def signal_handler(sig, frame):
            with open(processed_files_file, 'w') as file:
                file.write('\n'.join(processed_files))
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        concurrent.futures.wait(futures)

    with open(processed_files_file, 'w') as file:
        file.write('\n'.join(processed_files))


def calculate_edge_node_ratio(graph):
    edge_node_ratio = graph.number_of_edges() / graph.number_of_nodes()
    return edge_node_ratio


def update_features_with_ratio(graph_folder, feature_folder):
    entries_list = [os.path.join(graph_folder, entry) for entry in os.listdir(graph_folder)]

    with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(total=len(entries_list), desc='Progress') as pbar:
        futures = [executor.submit(update_file_with_ratio, entry, feature_folder, pbar) for entry in entries_list]

        def signal_handler(sig, frame):
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        concurrent.futures.wait(futures)


def update_file_with_ratio(graph_file, feature_folder, pbar):
    if graph_file.endswith('.tgf') or graph_file.endswith('.apx'):
        graph = parse_file(graph_file)
        graph_name = os.path.splitext(os.path.basename(graph_file))[0]

        edge_node_ratio = calculate_edge_node_ratio(graph)
        feature_file = os.path.join(feature_folder, f'{graph_name}.csv')
        if os.path.exists(feature_file):
            df = pd.read_csv(feature_file)
            df['edge_node_ratio'] = edge_node_ratio
            df.to_csv(feature_file, mode='w', header=True, index=False)
        pbar.update(1)



if __name__ == '__main__':
    # paths
    arg_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'
    output_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/processed_argumentation_frameworks'
    processed_files = '/Users/konraddrees/Documents/GitHub/sls-ml/files/processed_files.txt'
    preprocess_data(arg_folder, output_folder, processed_files)

    #update_features_with_ratio(arg_folder, output_folder)
