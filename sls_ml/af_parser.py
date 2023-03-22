import re
import networkx as nx



# A simple parser for the APX file format
def parse_apx(file_path):
    graph = nx.DiGraph()
    arg_pattern = re.compile(r'arg\((.+)\)')
    att_pattern = re.compile(r'att\((.+),\s*(.+)\)')

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            arg_match = arg_pattern.match(line)
            att_match = att_pattern.match(line)

            if arg_match:
                arg_name = arg_match.group(1)
                graph.add_node(arg_name)
            elif att_match:
                arg1, arg2 = att_match.group(1), att_match.group(2)
                graph.add_edge(arg1, arg2)

    return graph
