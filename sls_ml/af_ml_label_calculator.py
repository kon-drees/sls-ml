import csv
import math
import os

import concurrent.futures
import signal

import numpy as np
from tqdm import tqdm

from sls_ml.af_parser import parse_file
from sls_ml.af_util import get_random_labeling, get_mislabeled_args


def create_labels(graph):
    labels = {}
    iterations = calculate_iterations(graph)

    for _ in range(iterations):
        initial_labeling = get_random_labeling(set(graph.nodes()))

        mislabeled_args = get_mislabeled_args(initial_labeling, graph)

        for arg in mislabeled_args:
            labeling = initial_labeling.copy()
            flipped_labeling = labeling.copy()
            flipped_labeling[arg] = 'out' if flipped_labeling[arg] == 'in' else 'in'

            mislabeled_before_flip = len(mislabeled_args)
            mislabeled_after_flip = len(get_mislabeled_args(flipped_labeling, graph))

            stability_impact = mislabeled_before_flip - mislabeled_after_flip

            if arg not in labels:
                labels[arg] = {
                    "in": [],
                    "out": []
                }

            if initial_labeling[arg] == 'in':
                labels[arg]['in'].append(stability_impact)
            else:
                labels[arg]['out'].append(stability_impact)

    initial_labeling = {node: 'in' for node in graph.nodes()}

    mislabeled_args = get_mislabeled_args(initial_labeling, graph)

    for arg in mislabeled_args:
        labeling = initial_labeling.copy()
        flipped_labeling = labeling.copy()
        flipped_labeling[arg] = 'out' if flipped_labeling[arg] == 'in' else 'in'

        mislabeled_before_flip = len(mislabeled_args)
        mislabeled_after_flip = len(get_mislabeled_args(flipped_labeling, graph))

        stability_impact = mislabeled_before_flip - mislabeled_after_flip

        if arg not in labels:
            labels[arg] = {
                "in": [],
                "out": []
            }

        if initial_labeling[arg] == 'in':
            labels[arg]['in'].append(stability_impact)
        else:
            labels[arg]['out'].append(stability_impact)

    initial_labeling = {node: 'out' for node in graph.nodes()}

    mislabeled_args = get_mislabeled_args(initial_labeling, graph)

    for arg in mislabeled_args:
        labeling = initial_labeling.copy()
        flipped_labeling = labeling.copy()
        flipped_labeling[arg] = 'out' if flipped_labeling[arg] == 'in' else 'in'

        mislabeled_before_flip = len(mislabeled_args)
        mislabeled_after_flip = len(get_mislabeled_args(flipped_labeling, graph))

        stability_impact = mislabeled_before_flip - mislabeled_after_flip

        if arg not in labels:
            labels[arg] = {
                "in": [],
                "out": []
            }

        if initial_labeling[arg] == 'in':
            labels[arg]['in'].append(stability_impact)
        else:
            labels[arg]['out'].append(stability_impact)

    # Making sure that every node has at least one value
    for arg in graph.nodes():
        labeling = initial_labeling.copy()
        flipped_labeling = labeling.copy()
        flipped_labeling[arg] = 'out' if flipped_labeling[arg] == 'in' else 'in'

        mislabeled_before_flip = len(get_mislabeled_args(labeling, graph))
        mislabeled_after_flip = len(get_mislabeled_args(flipped_labeling, graph))

        stability_impact = mislabeled_before_flip - mislabeled_after_flip

        if arg not in labels:
            labels[arg] = {
                "in": [],
                "out": []
            }

        if initial_labeling[arg] == 'in':
            labels[arg]['in'].append(stability_impact)
        else:
            labels[arg]['out'].append(stability_impact)

    for arg in graph.nodes():
        initial_labeling[arg] = 'out' if initial_labeling[arg] == 'in' else 'in'



    for arg in graph.nodes():
        labeling = initial_labeling.copy()
        flipped_labeling = labeling.copy()
        flipped_labeling[arg] = 'out' if flipped_labeling[arg] == 'in' else 'in'

        mislabeled_before_flip = len(get_mislabeled_args(labeling, graph))
        mislabeled_after_flip = len(get_mislabeled_args(flipped_labeling, graph))

        stability_impact = mislabeled_before_flip - mislabeled_after_flip

        if arg not in labels:
            labels[arg] = {
                "in": [],
                "out": []
            }

        if initial_labeling[arg] == 'in':
            labels[arg]['in'].append(stability_impact)
        else:
            labels[arg]['out'].append(stability_impact)

    ml_labeling = {}
    print(labels)
    for arg in labels:
        stability_impacts_in = labels[arg]['in']
        stability_impacts_out = labels[arg]['out']

        # Calculate score for 'in'
        average_impacts_in = sum(stability_impacts_in) / len(stability_impacts_in)
        positive_impacts_in = [i for i in stability_impacts_in if i >= average_impacts_in]
        negative_impacts_in = [i for i in stability_impacts_in if i < average_impacts_in]
        score_in = len(positive_impacts_in) - len(negative_impacts_in)

        # Calculate score for 'out'
        average_impacts_out = sum(stability_impacts_out) / len(stability_impacts_out)
        positive_impacts_out = [i for i in stability_impacts_out if i >= average_impacts_out]
        negative_impacts_out = [i for i in stability_impacts_out if i < average_impacts_out]
        score_out = len(positive_impacts_out) - len(negative_impacts_out)

        # Decide whether to flip the labeling based on the scores
        majority_vote_in = 1 if score_in > 0 else 0
        majority_vote_out = 1 if score_out > 0 else 0

        ml_labeling[arg] = {
            "in": majority_vote_in,
            "out": majority_vote_out
        }

    return ml_labeling




def calculate_iterations(graph):
    num_nodes = len(graph.nodes())
    num_edges = len(graph.edges())

    # Determine the number of iterations based on the graph size
    iterations = int(math.log(num_nodes + num_edges + 1, 2)) * 100

    # Set a minimum number of iterations to ensure coverage
    min_iterations = 50

    # Return the maximum of the calculated iterations and the minimum iterations
    return max(iterations, min_iterations)


def labeling_data(argumentation_folder, output_folder, processed_files_file):
    processed_files = set()

    # Load the processed files if the file exists
    if os.path.exists(processed_files_file):
        with open(processed_files_file, 'r') as file:
            processed_files = set(file.read().splitlines())

    entries_list = [os.path.join(argumentation_folder, entry) for entry in os.listdir(argumentation_folder)]

    with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(total=len(entries_list), desc='Progress') as pbar:
        futures = []
        for entry in entries_list:
            futures.append(executor.submit(process_graph, entry, output_folder, processed_files, pbar))

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


def process_graph(file_path, output_folder, processed_files, pbar):
    if file_path.endswith('.tgf') or file_path.endswith('.apx'):
        graph = parse_file(file_path)
        graph_name = os.path.splitext(os.path.basename(file_path))[0]

        # Skip processing if the graph has already been processed
        if graph_name in processed_files:
            pbar.update(1)
            return

        # Calculate the labels using the create_labels function
        ml_labeling = create_labels(graph)

        # Save the ml_labeling to a CSV file
        output_file = os.path.join(output_folder, f'{graph_name}_labels.csv')
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Argument', 'In_Stability_Impact', 'Out_Stability_Impact']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for arg, data in ml_labeling.items():
                writer.writerow({
                    'Argument': arg,
                    'In_Stability_Impact': data['in'],
                    'Out_Stability_Impact': data['out']

                })

        # Add the processed graph to the set
        processed_files.add(graph_name)

        # Update the progress bar
        pbar.update(1)


if __name__ == '__main__':
    # paths
    output_label_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_argumentation_frameworks'
    output_label_for_visualization_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/visual_argumentation_frameworks'
    argumentation_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'

    processed_files = '/Users/konraddrees/Documents/GitHub/sls-ml/files/label_processed_files.txt'

    labeling_data(argumentation_folder, output_label_folder, processed_files)
