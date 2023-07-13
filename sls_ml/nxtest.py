import networkx as nx

# Create a directed graph
graph = nx.DiGraph()

# Add nodes
graph.add_nodes_from(['A', 'B', 'C', 'D'])

# Add directed edges
graph.add_edges_from([('B', 'A'), ('C', 'A'), ('A', 'D'), ('D', 'A')])

# Get predecessors of a node
print("Predecessors of 'A':", list(graph.predecessors('A')))
# Output: Predecessors of 'A': ['B', 'C']

# Get neighbors of a node
print("succ of 'A':", list(graph.successors('A')))
# Output: Neighbors of 'A': ['B', 'C', 'D']

# Get neighbors of a node
print("neighbors of 'A':", list(graph.neighbors('A')))
# Output: Neighbors of 'A': ['B', 'C', 'D']

