import unittest
import networkx as nx
from sls_ml.af_parser import parse_apx, parse_tgf, parse_file


class WalkAAFTest(unittest.TestCase):
    def test_parser_apx(self, ):
        # Given
        expected_af = nx.DiGraph()
        expected_af.add_nodes_from(['a1', 'a2', 'a3', 'a4'])
        expected_af.add_edges_from([('a1', 'a2'), ('a2', 'a3'), ('a3', 'a4'), ('a4', 'a3')])

        # When
        af = parse_file("testApxFile.apx")

        # Then
        self.assertEqual(set(af.nodes), set(expected_af.nodes))
        self.assertEqual(set(af.edges), set(expected_af.edges))

    def test_parser_tgf(self, ):
        # Given
        expected_af = nx.DiGraph()
        expected_af.add_nodes_from(['a1', 'a2', 'a3', 'a4'])
        expected_af.add_edges_from([('a1', 'a2'), ('a2', 'a3'), ('a3', 'a4'), ('a4', 'a3')])

        # When
        af = parse_file("testTgfFile.tgf")

        # Then
        self.assertEqual(set(af.nodes), set(expected_af.nodes))
        self.assertEqual(set(af.edges), set(expected_af.edges))


if __name__ == '__main__':
    unittest.main()
