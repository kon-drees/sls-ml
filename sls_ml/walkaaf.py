import warnings
warnings.simplefilter("ignore", category=UserWarning)
#models where trained with dataframes + featurenames, but i use numpy for speed
import random
import joblib
import networkx as nx
import numpy as np
import pandas as pd


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
                if labeling[random_argument] == 'in':
                    labeling[random_argument] = 'out'
                else:
                    labeling[random_argument] = 'in'

    return None



def get_initial_labeling_ml(args, features, model):
    arg_features = [features[arg] for arg in args]
    predictions = model.predict(arg_features)
    return {arg: 'in' if pred == 1 else 'out' for arg, pred in zip(args, predictions)}





def precompute_features(af_graph: nx.DiGraph):
    features_dict = {}
    betweenness_centrality = nx.betweenness_centrality(af_graph)
    closeness_centrality = nx.closeness_centrality(af_graph)
    predecessors_count = {node: len(list(af_graph.predecessors(node))) for node in af_graph.nodes()}
    pagerank = nx.pagerank(af_graph)

    for node in af_graph.nodes():
        features = [
            predecessors_count[node],
            closeness_centrality[node],
            betweenness_centrality[node],
            pagerank[node]
        ]
        features_dict[node] = features

    return features_dict


def choose_argument_ml(mislabeled, features_dict, model, labeling):
    data = []
    for arg in mislabeled:
        features = features_dict[arg]
        state = 1 if labeling[arg] == 'in' else 0
        data.append(features + [state])

    data_array = np.array(data)
    predictions = model.predict(data_array)

    candidates_to_flip = [arg for arg, pred in zip(mislabeled, predictions) if pred == 1]
    return candidates_to_flip


def walkaaf_with_ml1(af_graph: nx.DiGraph, flip_model, max_flips=2000, max_tries=200, g=0.5):
    args = set(af_graph.nodes)
    features_dict = precompute_features(af_graph)
    for current_try in range(max_tries):
        labeling = get_random_labeling(args)
        for current_flip in range(max_flips):
            mislabeled = get_mislabeled_args(labeling, af_graph)
            if len(mislabeled) == 0:
                return labeling
            else:
                if random.random() < g:
                    candidates_to_flip = choose_argument_ml(mislabeled, features_dict, flip_model, labeling)
                    if not candidates_to_flip:
                        chosen_argument = random.choice(mislabeled)
                    else:
                        chosen_argument = random.choice(candidates_to_flip)
                else:
                    chosen_argument = random.choice(mislabeled)
                if labeling[chosen_argument] == 'in':
                    labeling[chosen_argument] = 'out'
                else:
                    labeling[chosen_argument] = 'in'


    return None


def walkaaf_with_ml2(af_graph: nx.DiGraph, initial_model, max_flips=2000, max_tries=200, g=0.5):
    args = set(af_graph.nodes)
    features_dict = precompute_features(af_graph)

    for current_try in range(max_tries):

        if random.random() < g:
            labeling = get_initial_labeling_ml(args, features_dict, initial_model)
        else:
            labeling = get_random_labeling(args)

        for current_flip in range(max_flips):
            mislabeled = get_mislabeled_args(labeling, af_graph)
            if len(mislabeled) == 0:
                return labeling  # in_labels
            else:
                random_argument = random.choice(mislabeled)
                if labeling[random_argument] == 'in':
                    labeling[random_argument] = 'out'
                else:
                    labeling[random_argument] = 'in'
    return None


def walkaaf_with_ml3(af_graph: nx.DiGraph, flip_model, initial_model, max_flips=2000, max_tries=200, g=0.5):
    args = set(af_graph.nodes)
    features_dict = precompute_features(af_graph)

    for current_try in range(max_tries):
        if random.random() < g:
            labeling = get_initial_labeling_ml(args, features_dict, initial_model)
        else:
            labeling = get_random_labeling(args)
        for current_flip in range(max_flips):
            mislabeled = get_mislabeled_args(labeling, af_graph)
            if len(mislabeled) == 0:
                return labeling
            else:
                if random.random() < g:
                    candidates_to_flip = choose_argument_ml(mislabeled, features_dict, flip_model, labeling)
                    if not candidates_to_flip:
                        chosen_argument = random.choice(mislabeled)
                    else:
                        chosen_argument = random.choice(candidates_to_flip)
                else:
                    chosen_argument = random.choice(mislabeled)
                if labeling[chosen_argument] == 'in':
                    labeling[chosen_argument] = 'out'
                else:
                    labeling[chosen_argument] = 'in'
    return None