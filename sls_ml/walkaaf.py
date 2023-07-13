import random

import joblib
import networkx as nx
import numpy as np


# create a random labeling
def get_random_labeling(args):
    return {arg: random.choice(['in', 'out']) for arg in args}


# checks if a labeling is stable for a given af graph
def is_stable(labeling, af_graph: nx.DiGraph):
    for arg in af_graph.nodes():
        if labeling[arg] == 'out':
            in_attackers = [attacker for attacker in af_graph.predecessors(arg) if labeling[attacker] == 'in']
            if not in_attackers:
                return False
        elif labeling[arg] == 'in':
            if any(labeling[u] != 'out' for u in af_graph.predecessors(arg)) or any(
                    labeling[v] != 'out' for v in af_graph.neighbors(arg)):
                return False
    return True


# returns mislabeled arguments
def get_mislabeled_args(labeling, af_graph: nx.DiGraph):
    mislabeled = []
    for arg in af_graph.nodes():
        if labeling[arg] == 'out':
            in_attackers = [attacker for attacker in af_graph.predecessors(arg) if labeling[attacker] == 'in']
            if not in_attackers:
                mislabeled.append(arg)
        elif labeling[arg] == 'in':
            if any(labeling[u] != 'out' for u in af_graph.predecessors(arg)) or any(
                    labeling[v] != 'out' for v in af_graph.neighbors(arg)):
                mislabeled.append(arg)
    return mislabeled




# WalkAAF Algorithm returns a single stable extension and None if no labeling was found
def walkaaf(af_graph: nx.DiGraph, max_flips=2000, max_tries=200):
    args = set(af_graph.nodes)
    for current_try in range(max_tries):
        labeling = get_random_labeling(args)
        for current_flip in range(max_flips):
            mislabeled = get_mislabeled_args(labeling, af_graph)
            if len(mislabeled) == 0:
                # in_labels = {arg for arg, label in labeling.items() if label == 'in'}
                return labeling  # in_labels
            else:
                random_argument = random.choice(mislabeled)
                labeling[random_argument] = 'out' if labeling[random_argument] == 'in' else 'out'

    return None


def extract_features(graph):
    features = []
    for arg in graph.nodes():
        in_neighbors = len(list(graph.predecessors(arg)))
        out_neighbors = len(list(graph.successors(arg)))
        second_order_in_neighbors = sum(
            [len(list(graph.predecessors(neighbor))) for neighbor in graph.predecessors(arg)])
        second_order_out_neighbors = sum([len(list(graph.successors(neighbor))) for neighbor in graph.successors(arg)])
        features.append([in_neighbors, out_neighbors, second_order_in_neighbors, second_order_out_neighbors])
    return features


def extract_features_from_arg(graph, arg):
    features = [[len(list(graph.predecessors(arg))),
                 len(list(graph.successors(arg))),
                 sum(len(list(graph.predecessors(neighbor))) for neighbor in graph.predecessors(arg)),
                 sum(len(list(graph.successors(neighbor))) for neighbor in graph.successors(arg))]
                for arg in graph.nodes()]
    return features


def walkaaf_ml(af_graph: nx.DiGraph, model_path='ml_model.joblib', max_flips=2000, max_tries=200,  G=0.8):
    args = list(af_graph.nodes)

    # Load the trained model
    model = joblib.load(model_path)

    # Extract features for all arguments once
    features_dict = {arg: extract_features_from_arg(af_graph, arg) for arg in args}

    for current_try in range(max_tries):
        labeling = get_random_labeling(args)
        for current_flip in range(max_flips):
            mislabeled = get_mislabeled_args(labeling, af_graph)
            if not mislabeled:
                return labeling
            else:
                if random.random() < G:
                    # Use pre-computed features for mislabeled arguments
                    mislabeled_features = np.array([features_dict[arg] for arg in mislabeled]).reshape(-1, 4)

                    predicted_labels = np.array(model.predict_proba(mislabeled_features))

                    # Choose the argument with the highest probability of decreasing the number of mislabeled arguments
                    arg_to_flip_idx = np.argmax(predicted_labels[:, 0])
                    arg_to_flip = mislabeled[arg_to_flip_idx]
                else:
                    # Randomly choose an argument to flip
                    arg_to_flip = random.choice(mislabeled)
                # Flip the label of the chosen argument
                labeling[arg_to_flip] = 'out' if labeling[arg_to_flip] == 'in' else 'in'

    return None




