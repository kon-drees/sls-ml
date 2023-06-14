import os
import pandas as pd
import concurrent.futures
import signal
import networkx as nx
from tqdm import tqdm


from sls_ml.af_parser import parse_file
from sls_ml.af_util import extract_features_graph_arg


def process_graph(file_path, output_folder, processed_files, pbar):
    if file_path.endswith('.tgf') or file_path.endswith('.apx'):
        graph = parse_file(file_path)
        graph_name = os.path.splitext(os.path.basename(file_path))[0]

        # Skip processing if the graph has already been processed
        if graph_name in processed_files:
            return

        # Collect features for all arguments
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

        # Save the DataFrame to a CSV file for the argumentation framework
        output_file = os.path.join(output_folder, f'{graph_name}.csv')
        data.to_csv(output_file, mode='w', header=True, index=False)

        # Add the processed graph to the set
        processed_files.add(graph_name)

        # Update the progress bar
        pbar.update(1)


def preprocess_data(directory_path, output_folder, processed_files_file):
    processed_files = set()

    # Load the processed files if the file exists
    if os.path.exists(processed_files_file):
        with open(processed_files_file, 'r') as file:
            processed_files = set(file.read().splitlines())

    entries_list = [os.path.join(directory_path, entry) for entry in os.listdir(directory_path)]

    with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(total=len(entries_list), desc='Progress') as pbar:
        futures = [executor.submit(process_graph, entry, output_folder, processed_files, pbar) for entry in entries_list]

        # Add a signal handler to capture the interrupt signal (Ctrl+C)
        def signal_handler(sig, frame):
            # Save the updated processed files
            with open(processed_files_file, 'w') as file:
                file.write('\n'.join(processed_files))
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Wait for all the futures to complete
        concurrent.futures.wait(futures)

    # Save the updated processed files
    with open(processed_files_file, 'w') as file:
        file.write('\n'.join(processed_files))


if __name__ == '__main__':
    # paths
    directory_path = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'
    output_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/processed_argumentation_frameworks'
    processed_files = '/Users/konraddrees/Documents/GitHub/sls-ml/files/processed_files.txt'

    preprocess_data(directory_path, output_folder, processed_files)
