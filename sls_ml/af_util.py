import networkx as nx
import pandas as pd
import random


# extract features for the input graph and argument
def extract_features_graph_arg(graph, arg):
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
        nx.pagerank(graph)[arg]
    ]
    return features


def extract_features_graph(graph):
    features = [
        graph.number_of_nodes(),
        graph.number_of_edges(),

    ]
    return features


def read_features_csv_file(file_path):
    data = pd.read_csv(file_path)
    return data


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

def get_mislabeled_args(labeling, af_graph: nx.DiGraph):
    mislabeled = []
    for arg in af_graph.nodes():
        if labeling[arg] == 'out':
            in_attackers = [attacker for attacker in af_graph.predecessors(arg) if labeling[attacker] == 'in']
            if not in_attackers:
                mislabeled.append(arg)
        elif labeling[arg] == 'in':
            out_attackers = [attacker for attacker in af_graph.predecessors(arg) if labeling[attacker] == 'in']
            in_attackers = [attacker for attacker in af_graph.successors(arg) if labeling[attacker] == 'in']
            if out_attackers or in_attackers:
                mislabeled.append(arg)
    return mislabeled
# returns mislabeled arguments
def get_mislabeled_args_2(labeling, af_graph: nx.DiGraph):
    mislabeled = []
    for arg in af_graph.nodes():
        if labeling[arg] == 'out':
            in_attackers = [attacker for attacker in af_graph.predecessors(arg) if labeling[attacker] == 'in']
            if not in_attackers:
                mislabeled.append(arg)
        elif labeling[arg] == 'in':
            if any(labeling[u] != 'out' for u in af_graph.predecessors(arg)) or any(
                    labeling[v] != 'out' for v in af_graph.successors(arg)):
                mislabeled.append(arg)
    return mislabeled
