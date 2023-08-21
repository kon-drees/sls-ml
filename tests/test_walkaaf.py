import os
import sys
import warnings

import torch

from sls_ml.af_nn_model_creator import AAF_GCNConv

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.utils._testing import ignore_warnings

import unittest
import networkx as nx
from joblib import load
from sls_ml.walkaaf import walkaaf, is_stable, walkaaf_with_ml1, walkaaf_with_ml2, walkaaf_with_ml3, walkaaf_with_ml3_nn


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

    @ignore_warnings(category=UserWarning)
    def test_walkaaf_ml(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
        G.add_edges_from([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'e'), ('e', 'd'), ('d', 'c')])

        model = load('/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models/trained_model_RandomForest_rn_red.joblib')

        # When
        stable_extension = walkaaf_with_ml1(G, model, g=0.2)

        # Then
        # Two cases because there exists one stable extension but the algorithm can just find one
        if stable_extension:
            print(stable_extension)
            self.assertTrue(is_stable(stable_extension, G))
        else:
            print("no")
            self.assertIsNone(stable_extension)

    @ignore_warnings(category=UserWarning)
    def test_walkaaf_ml2(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
        G.add_edges_from([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'e'), ('e', 'd'), ('d', 'c')])


        model_in = load('/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models/trained_model_RandomForest_in_red.joblib')

        # When
        stable_extension = walkaaf_with_ml2(G, model_in, g=0.2)

        # Then
        # Two cases because there exists one stable extension but the algorithm can just find one
        if stable_extension:
            print(stable_extension)
            self.assertTrue(is_stable(stable_extension, G))
        else:
            print("no")
            self.assertIsNone(stable_extension)

    @ignore_warnings(category=UserWarning)
    def test_walkaaf_ml3(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
        G.add_edges_from([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'e'), ('e', 'd'), ('d', 'c')])

        model_rn = load(
            '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models/trained_model_RandomForest_rn_red.joblib')
        model_in = load(
            '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models/trained_model_RandomForest_in_red.joblib')
        # When

        stable_extension = walkaaf_with_ml3(G, model_rn, model_in, g=0.2)

        # Then
        # Two cases because there exists one stable extension but the algorithm can just find one
        if stable_extension:
            print(stable_extension)
            self.assertTrue(is_stable(stable_extension, G))
        else:
            print("no")
            self.assertIsNone(stable_extension)



    def test_walkaaf_ml4_nn(self):
        # Given
        G = nx.DiGraph()
        G.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
        G.add_edges_from([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'e'), ('e', 'd'), ('d', 'c')])

        model_in_red = AAF_GCNConv(4, 2)

        output_folder = "/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models"  # Replace this with your actual path
        PATH = os.path.join(output_folder, "g_nn_in_red.pt")
        model_in_red.load_state_dict(torch.load(PATH))
        model_in_red.eval()

        model_rn_red = AAF_GCNConv(5, 2)
        PATH = os.path.join(output_folder, "g_nn_rn_red.pt")
        model_rn_red.load_state_dict(torch.load(PATH))
        model_rn_red.eval()



        stable_extension = walkaaf_with_ml3_nn(G, model_rn_red, model_in_red, g=0.8)

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
