import csv
import math
import os
import random

import concurrent.futures
import signal
import threading

import networkx as nx
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


# Determine the number of iterations based on the graph size
def calculate_iterations(graph):
    num_nodes = len(graph.nodes())
    num_edges = len(graph.edges())
    iterations = int(math.log(num_nodes + num_edges + 1, 2)) * 30
    min_iterations = 50
    return max(iterations, min_iterations)


def process_graph(file_path, output_folder, processed_files, lock, pbar, executor_id):
    if file_path.endswith('.tgf') or file_path.endswith('.apx'):
        graph = parse_file(file_path)
        graph_name = os.path.splitext(os.path.basename(file_path))[0]

        with lock:
            if graph_name in processed_files:
                pbar.update()
                return
            processed_files.add(graph_name)


        ml_labeling = create_labels(graph)

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

        pbar.update()


def labeling_data_random(argumentation_folder, output_folder, processed_files_file):
    # Load the processed files if the file exists
    if os.path.exists(processed_files_file):
        with open(processed_files_file, 'r') as file:
            processed_files = set(file.read().splitlines())
    else:
        processed_files = set()

    entries_list = [os.path.join(argumentation_folder, entry) for entry in os.listdir(argumentation_folder)]

    total = len(entries_list)
    pbar = tqdm(total=total, desc='Overall Progress')
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, entry in enumerate(entries_list):
            futures.append(executor.submit(process_graph, entry, output_folder, processed_files, lock, pbar, i))

        def signal_handler(sig, frame):
            # Save the updated processed files
            with open(processed_files_file, 'w') as file:
                file.write('\n'.join(processed_files))
            pbar.close()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        concurrent.futures.wait(futures)
    pbar.close()
    with open(processed_files_file, 'w') as file:
        file.write('\n'.join(processed_files))


# returns the best labeling within its tries and flips
def walkaaf_labeling(af_graph: nx.DiGraph, max_flips=1000, max_tries=100):
    args = set(af_graph.nodes)
    best_labels = 0
    best_labeling = None
    for current_try in range(max_tries):
        labeling = get_random_labeling(args)
        for current_flip in range(max_flips):
            mislabeled = get_mislabeled_args(labeling, af_graph)
            if len(mislabeled) == 0:
                return labeling
            else:
                correct_labels = len(args) - len(mislabeled)
                if correct_labels > best_labels:
                    best_labels = correct_labels
                    best_labeling = labeling
                random_argument = random.choice(mislabeled)
                labeling[random_argument] = 'out' if labeling[random_argument] == 'in' else 'in'
    return best_labeling


def process_graph_initial(file_path, output_folder, processed_files, executor_id, pbar, lock):
    if file_path.endswith('.tgf') or file_path.endswith('.apx'):
        graph = parse_file(file_path)
        graph_name = os.path.splitext(os.path.basename(file_path))[0]
        if graph_name in processed_files:
            with lock:
                pbar.update()
            return

        ml_labeling = walkaaf_labeling(graph)
        output_file = os.path.join(output_folder, f'{graph_name}_labels.csv')
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Argument', 'Label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for key, value in sorted(ml_labeling.items()):
                writer.writerow({
                    'Argument': key,
                    'Label': value
                })
        processed_files.add(graph_name)
        with lock:
            pbar.update()


def labeling_data_inital(argumentation_folder, output_label_folder, processed_files_file):
    processed_files = set()
    if os.path.exists(processed_files_file):
        with open(processed_files_file, 'r') as file:
            processed_files = set(file.read().splitlines())
    entries_list = [os.path.join(argumentation_folder, entry) for entry in os.listdir(argumentation_folder)]
    total = len(entries_list)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        lock = threading.Lock()
        pbar = tqdm(total=total, desc='Processing progress')
        futures = []
        for i, entry in enumerate(entries_list):
            futures.append(executor.submit(process_graph_initial, entry, output_label_folder, processed_files, i, pbar, lock))

        def signal_handler(sig, frame):
            with open(processed_files_file, 'w') as file:
                file.write('\n'.join(processed_files))
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        concurrent.futures.wait(futures)

    pbar.close()
    with open(processed_files_file, 'w') as file:
        file.write('\n'.join(processed_files))


if __name__ == '__main__':
    # paths
    output_label_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_argumentation_frameworks'
    output_label_initial_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_initial_argumentation_frameworks'
    output_label_for_visualization_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/visual_argumentation_frameworks'
    argumentation_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'

    processed_files = '/Users/konraddrees/Documents/GitHub/sls-ml/files/label_processed_files.txt'
    processed_initial_files = '/Users/konraddrees/Documents/GitHub/sls-ml/files/label_initial_processed_files.txt'

    labeling_data_random(argumentation_folder, output_label_folder, processed_files)
    labeling_data_inital(argumentation_folder, output_label_initial_folder, processed_initial_files)
