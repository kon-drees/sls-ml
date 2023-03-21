import networkx as nx


# A simple parser for the APX file format
def parse_apx(file_path):
    af = nx.DiGraph()

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('arg('):
                node = line[4:-1]
                af.add_node(node)
            elif line.startswith('att('):
                attacker, attacked = line[4:-1].split(',')
                af.add_edge(attacker, attacked)

    return af

