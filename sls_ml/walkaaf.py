import random
import networkx as nx


# create a random labeling
def get_random_labeling(args):
    labeling = {}
    for arg in args:
        labeling[arg] = random.choice(['in', 'out'])
    return labeling


# checks if a labeling is stable for a given af graph
def is_stable(labeling, af_graph: nx.DiGraph):
    for arg in af_graph.nodes():
        if labeling[arg] == 'in':
            for v in af_graph.neighbors(arg):
                if labeling[v] != 'out':
                    return False
        elif labeling[arg] == 'out':
            # get the attackers of the argument (in-label) neighbours  only get the successors(attackers) of a node
            in_attackers = [attacker for attacker in af_graph.predecessors(arg) if labeling[attacker] == 'in']
            if not in_attackers:
                return False
        elif labeling[arg] == 'undec':
            return False
    return True


# returns mislabeled arguments
def get_mislabeled_args(labeling, af_graph: nx.DiGraph):
    mislabeled = []
    for arg in af_graph.nodes():
        if labeling[arg] == 'in':
            for v in af_graph.neighbors(arg):
                if labeling[v] != 'out':
                    mislabeled.append(arg)
                    break
        elif labeling[arg] == 'out':
            # get the attackers of the argument (in-label) neighbours  only get the successors(attackers) of a node
            in_attackers = [attacker for attacker in af_graph.predecessors(arg) if labeling[attacker] == 'in']
            if not in_attackers:
                mislabeled.append(arg)
        elif labeling[arg] == 'undec':
            mislabeled.append(arg)
    return None if not mislabeled else mislabeled


# WalkAAF Algorithm returns a single stable extension and None if no labeling was found
def walkaaf(af_graph: nx.DiGraph, max_flips=2000, max_tries=200):
    args = set(af_graph.nodes)
    for current_try in range(max_tries):
        labeling = get_random_labeling(args)
        for current_flip in range(max_flips):
            mislabeled = get_mislabeled_args(labeling, af_graph)
            if mislabeled is None:
                # in_labels = {arg for arg, label in labeling.items() if label == 'in'}
                return labeling  # in_labels
            else:
                random_argument = random.choice(mislabeled)
                labeling[random_argument] = 'out' if labeling[random_argument] == 'in' else 'out'

    return None

