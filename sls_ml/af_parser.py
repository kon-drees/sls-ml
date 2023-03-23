import os
import re
import networkx as nx


# A simple parser for the APX file format using regex
def parse_apx(file_path):
    aaf_graph = nx.DiGraph()
    # regex
    argument_pattern = re.compile(r'arg\((.+)\)')
    attack_pattern = re.compile(r'att\((.+),\s*(.+)\)')

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            argument_match = argument_pattern.match(line)
            attack_match = attack_pattern.match(line)

            if argument_match:
                arg_name = argument_match.group(1)
                aaf_graph.add_node(arg_name)
            elif attack_match:
                arg1, arg2 = attack_match.group(1), attack_match.group(2)
                aaf_graph.add_edge(arg1, arg2)

    return aaf_graph


# A simple parser for the tgf file format using manual splitting
def parse_tgf(file_path):
    aaf_graph = nx.DiGraph()
    with open(file_path, 'r') as file:
        is_argument = True
        for line in file:
            line = line.strip()
            if line == "#":
                is_argument = False
                continue
            if is_argument:
                argument = line.split()[0]
                aaf_graph.add_node(argument)
            else:
                source, target = line.split()
                aaf_graph.add_edge(source, target)

    return aaf_graph


def parse_file(file_path):
    _, file_ending = os.path.splitext(file_path)
    if file_ending.lower() == '.tgf':
        return parse_tgf(file_path)
    elif file_ending.lower() == '.apx':
        return parse_apx(file_path)
    else:
        raise ValueError(f"Unsupported file: {file_ending}")
