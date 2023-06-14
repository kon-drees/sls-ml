import networkx as nx


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