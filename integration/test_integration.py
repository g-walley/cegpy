# from pathlib import Path
# import networkx as nx
# import pandas as pd
# import os.path
# from src.cegpy.graphs.ceg import ChainEventGraph
# from src.cegpy.graphs.ceg import Evidence
# from src.cegpy.trees.staged import StagedTree
# from src.cegpy.utilities.util import Util


# class TestReducedOutputs:
#     def setup(self):
#         G = nx.MultiDiGraph()
#         nodes = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5',
#                  'w6', 'w7', 'w8', 'w&infin;']
#         G.add_edge('w0', 'w1', 'a', probability=0.3)
#         G.add_edge('w0', 'w1', 'b', probability=0.3)
#         G.add_edge('w0', 'w5', 'j', probability=0.2)
#         G.add_edge('w0', 'w7', 'k', probability=0.05)
#         G.add_edge('w0', 'w6', 'l', probability=0.05)
#         G.add_edge('w0', 'w3', 's', probability=0.1)

#         G.add_edge('w1', 'w2', 'c', probability=0.88)
#         G.add_edge('w1', 'w3', 'd', probability=0.12)

#         G.add_edge('w2', 'w4', 'e', probability=0.95)
#         G.add_edge('w2', 'w4', 'f', probability=0.05)

#         G.add_edge('w3', 'w4', 'g', probability=0.5)
#         G.add_edge('w3', 'w8', 'h', probability=0.5)

#         G.add_edge('w4', 'w8', 'i', probability=0.9)
#         G.add_edge('w4', 'w&infin;', 'r', probability=0.1)

#         G.add_edge('w5', 'w6', 'm', probability=0.6)
#         G.add_edge('w5', 'w7', 'n', probability=0.4)

#         G.add_edge('w6', 'w7', 'o', probability=1)

#         G.add_edge('w7', 'w8', 'p', probability=1)

#         G.add_edge('w8', 'w&infin;', 'q', probability=1)

#         G.add_nodes_from(nodes)
#         H = ChainEventGraph(G)
#         self.evidence = Evidence(H)
#         print(self.evidence)

#     def test_subgraph(self):
#         def update_path_lists():
#             self.evidence._Evidence__graph._ChainEventGraph__update_path_list()
#             self.evidence._Evidence__update_path_list()

#         fname = Util.create_path('out/Evidence_pre_test', False, 'pdf')
#         self.evidence._Evidence__graph.create_figure(fname)
#         update_path_lists()
#         self.evidence.add_edge('w8', 'w&infin;', 'q', Evidence.CERTAIN)
#         self.evidence.add_node('w4', Evidence.CERTAIN)
#         reduced = self.evidence.reduced_graph

#         expected_reduced_probabilities = {
#             ('w0', 'w1', 'a'): 0.46,
#             ('w0', 'w1', 'b'): 0.46,
#             ('w0', 'w3', 's'): 0.08,
#             ('w1', 'w2', 'c'): 0.94,
#             ('w1', 'w3', 'd'): 0.06,
#             ('w3', 'w4', 'g'): 1.0,
#             ('w2', 'w4', 'e'): 0.95,
#             ('w2', 'w4', 'f'): 0.05,
#             ('w4', 'w8', 'i'): 1.0,
#             ('w8', 'w&infin;', 'q'): 1.0
#         }
#         actual_reduced_probabilities = nx.get_edge_attributes(
#             reduced,
#             'probability'
#         )

#         for edge, probability in expected_reduced_probabilities.items():
#             try:
#                 actual = round(actual_reduced_probabilities[edge], 2)
#                 assert actual == probability
#             except KeyError:
#                 assert False  # edge doesnt exist in graph!

#         fname = Util.create_path('out/Evidence_test_certain', False, 'pdf')
#         reduced.create_figure(fname)

#         expected_paths = [
#             [('w0', 'w1', 'a'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'a'), ('w1', 'w2', 'c'), ('w2', 'w4', 'f'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'b'), ('w1', 'w2', 'c'), ('w2', 'w4', 'f'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'b'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w3', 's'), ('w3', 'w4', 'g'), ('w4', 'w8', 'i'),
#                 ('w8', 'w&infin;', 'q')]
#         ]
#         for path in expected_paths:
#             assert path in self.evidence.path_list
#         assert len(expected_paths) == len(self.evidence.path_list)

#         self.evidence.remove_edge('w8', 'w&infin;', 'q', Evidence.CERTAIN)
#         self.evidence.remove_node('w4', Evidence.CERTAIN)
#         update_path_lists()
#         self.evidence.add_node('w6', Evidence.UNCERTAIN)
#         self.evidence.add_node('w3', Evidence.UNCERTAIN)
#         reduced = self.evidence.reduced_graph

#         expected_reduced_probabilities = {
#             ('w0', 'w1', 'a'): 0.105,
#             ('w0', 'w1', 'b'): 0.105,
#             ('w0', 'w5', 'j'): 0.351,
#             ('w0', 'w6', 'l'): 0.146,
#             ('w0', 'w3', 's'): 0.292,
#             ('w1', 'w3', 'd'): 1.0,
#             ('w5', 'w6', 'm'): 1.0,
#             ('w7', 'w8', 'p'): 1.0,
#             ('w6', 'w7', 'o'): 1.0,
#             ('w3', 'w4', 'g'): 0.5,
#             ('w3', 'w8', 'h'): 0.5,
#             ('w4', 'w8', 'i'): 0.9,
#             ('w4', 'w&infin;', 'r'): 0.1,
#             ('w8', 'w&infin;', 'q'): 1.0
#         }
#         actual_reduced_probabilities = nx.get_edge_attributes(
#             reduced,
#             'probability'
#         )

#         for edge, probability in expected_reduced_probabilities.items():
#             try:
#                 actual = round(actual_reduced_probabilities[edge], 3)
#                 assert actual == probability
#             except KeyError:
#                 assert False  # edge doesnt exist in graph!

#         fname = Util.create_path('out/Evidence_test_uncertain', False, 'pdf')
#         reduced.create_figure(fname)

#         expected_paths = [
#             [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
#                 ('w4', 'w&infin;', 'r')],
#             [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
#                 ('w4', 'w&infin;', 'r')],
#             [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w8', 'h'),
#                 ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w8', 'h'),
#                 ('w8', 'w&infin;', 'q')],
#             [('w0', 'w5', 'j'), ('w5', 'w6', 'm'), ('w6', 'w7', 'o'),
#                 ('w7', 'w8', 'p'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w6', 'l'), ('w6', 'w7', 'o'), ('w7', 'w8', 'p'),
#                 ('w8', 'w&infin;', 'q')],
#             [('w0', 'w3', 's'), ('w3', 'w4', 'g'), ('w4', 'w8', 'i'),
#                 ('w8', 'w&infin;', 'q')],
#             [('w0', 'w3', 's'), ('w3', 'w4', 'g'), ('w4', 'w&infin;', 'r')],
#             [('w0', 'w3', 's'), ('w3', 'w8', 'h'), ('w8', 'w&infin;', 'q')]
#         ]
#         for path in expected_paths:
#             assert path in self.evidence.path_list
#         assert len(expected_paths) == len(self.evidence.path_list)

#         self.evidence.remove_node('w3', Evidence.UNCERTAIN)
#         self.evidence.remove_node('w6', Evidence.UNCERTAIN)
#         update_path_lists()
#         self.evidence.add_edge('w2', 'w4', 'e', Evidence.UNCERTAIN)
#         self.evidence.add_edge('w3', 'w4', 'g', Evidence.UNCERTAIN)
#         self.evidence.add_edge('w3', 'w8', 'h', Evidence.UNCERTAIN)
#         self.evidence.add_node('w1', Evidence.CERTAIN)

#         reduced = self.evidence.reduced_graph

#         expected_reduced_probabilities = {
#             ('w0', 'w1', 'a'): 0.5,
#             ('w0', 'w1', 'b'): 0.5,
#             ('w1', 'w2', 'c'): 0.87,
#             ('w1', 'w3', 'd'): 0.13,
#             ('w3', 'w4', 'g'): 0.5,
#             ('w3', 'w8', 'h'): 0.5,
#             ('w2', 'w4', 'e'): 1.0,
#             ('w4', 'w8', 'i'): 0.9,
#             ('w4', 'w&infin;', 'r'): 0.1,
#             ('w8', 'w&infin;', 'q'): 1.0
#         }
#         actual_reduced_probabilities = nx.get_edge_attributes(
#             reduced,
#             'probability'
#         )

#         for edge, probability in expected_reduced_probabilities.items():
#             try:
#                 actual = round(actual_reduced_probabilities[edge], 2)
#                 assert actual == probability
#             except KeyError:
#                 assert False  # edge doesnt exist in graph!

#         fname = Util.create_path('out/Evidence_test_certain_and_uncertain', False, 'pdf')
#         reduced.create_figure(fname)

#         expected_paths = [
#             [('w0', 'w1', 'a'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
#                 ('w4', 'w&infin;', 'r')],
#             [('w0', 'w1', 'a'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'b'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
#                 ('w4', 'w&infin;', 'r')],
#             [('w0', 'w1', 'b'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w8', 'h'),
#                 ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
#                 ('w4', 'w&infin;', 'r')],
#             [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w8', 'h'),
#                 ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
#                 ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
#             [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
#                 ('w4', 'w&infin;', 'r')],
#         ]
#         for path in expected_paths:
#             assert path in self.evidence.path_list
#         assert len(expected_paths) == len(self.evidence.path_list)


# class TestCegOutput:
#     def setup(self):
#         df_path = Path(__file__).resolve(
#             ).parent.parent.joinpath(
#             'data/medical_dm_modified.xlsx')

#         st = StagedTree(
#             dataframe=pd.read_excel(df_path),
#             name="medical_staged"
#         )
#         st.calculate_AHC_transitions()
#         self.ceg = ChainEventGraph(st)

#     def test_creation_of_ceg(self) -> None:
#         self.ceg.generate()
#         fname = Util.create_path('out/medical_dm_CEG', True, 'pdf')
#         self.ceg.create_figure(fname)
#         assert os.path.isfile(fname)

#     def test_generation(self) -> None:
#         self.ceg.generate()
#         path_list = self.ceg.path_list
#         nodes = ['w0', 'w1', 'w4', 'w10', 'w9', 'w&infin;']

#         subgraph = self.ceg.subgraph(nodes).copy()
#         fname = Util.create_path('out/Subgraph_test', True, 'pdf')
#         subgraph.create_figure(fname)
#         print(path_list)


# class TestCegNonStrat:
#     def setup(self):
#         df_path = Path(__file__).resolve().parent.parent.joinpath(
#             'data/Falls_Data.xlsx'
#         )
#         st = StagedTree(
#             dataframe=pd.read_excel(df_path),
#             name='Falls Staged Tree'
#         )
#         st.calculate_AHC_transitions()
#         self.ceg = ChainEventGraph(st)

#     def test_generation(self):
#         self.ceg.generate()
