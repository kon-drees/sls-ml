import unittest
import networkx as nx
from sls_ml.walkaaf import walkaaf, is_stable


class TestWalkAAf(unittest.TestCase):

    def test_walkaaf(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
        G.add_edges_from([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'e'), ('e', 'd'), ('d', 'c')])

        # When
        stable_extension = walkaaf(G)

        # Then
        # Two cases because there exists one stable extension but the algorithm can just find one
        if stable_extension:
            print(stable_extension)
            self.assertTrue(is_stable(stable_extension, G))
        else:
            print("no")
            self.assertIsNone(stable_extension)


if __name__ == '__main__':
    unittest.main()
