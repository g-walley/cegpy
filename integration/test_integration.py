from ..src.cegpy.graphs.ceg import ChainEventGraph
from ..src.cegpy.graphs.ceg import Evidence
from ..src.cegpy.trees.staged import StagedTree
from ..src.cegpy.utilities.util import Util
from pathlib import Path
import networkx as nx
import pandas as pd
import os.path


class TestReducedOutputs:
    def setup(self):
        G = nx.MultiDiGraph()
        nodes = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5',
                 'w6', 'w7', 'w8', 'w&infin;']
        edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w1', 'b'),
            ('w1', 'w2', 'c'),
            ('w1', 'w3', 'd'),
            ('w2', 'w4', 'e'),
            ('w2', 'w4', 'f'),
            ('w3', 'w4', 'g'),
            ('w3', 'w8', 'h'),
            ('w4', 'w8', 'i'),
            ('w0', 'w5', 'j'),
            ('w0', 'w7', 'k'),
            ('w0', 'w6', 'l'),
            ('w5', 'w6', 'm'),
            ('w5', 'w7', 'n'),
            ('w6', 'w7', 'o'),
            ('w7', 'w8', 'p'),
            ('w8', 'w&infin;', 'q'),
            ('w4', 'w&infin;', 'r'),
            ('w0', 'w3', 's')
        ]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        H = ChainEventGraph(G)
        self.evidence = Evidence(H)
        print(self.evidence)

    def test_subgraph(self):
        def update_path_lists():
            self.evidence._Evidence__graph._ChainEventGraph__update_path_list()
            self.evidence._Evidence__update_path_list()

        fname = Util.create_path('out/Evidence_pre_test', False, 'pdf')
        self.evidence._Evidence__graph.create_figure(fname)
        update_path_lists()
        self.evidence.add_edge('w8', 'w&infin;', 'q', Evidence.CERTAIN)
        self.evidence.add_vertex('w4', Evidence.CERTAIN)
        reduced = self.evidence.reduced_graph
        fname = Util.create_path('out/Evidence_test_certain', False, 'pdf')
        reduced.create_figure(fname)

        expected_paths = [
            [('w0', 'w1', 'a'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'a'), ('w1', 'w2', 'c'), ('w2', 'w4', 'f'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'b'), ('w1', 'w2', 'c'), ('w2', 'w4', 'f'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'b'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w3', 's'), ('w3', 'w4', 'g'), ('w4', 'w8', 'i'),
                ('w8', 'w&infin;', 'q')]
        ]
        for path in expected_paths:
            assert path in self.evidence.path_list
        assert len(expected_paths) == len(self.evidence.path_list)

        self.evidence.remove_edge('w8', 'w&infin;', 'q', Evidence.CERTAIN)
        self.evidence.remove_vertex('w4', Evidence.CERTAIN)
        update_path_lists()
        self.evidence.add_vertex('w6', Evidence.UNCERTAIN)
        self.evidence.add_vertex('w3', Evidence.UNCERTAIN)
        reduced = self.evidence.reduced_graph
        fname = Util.create_path('out/Evidence_test_uncertain', False, 'pdf')
        reduced.create_figure(fname)

        expected_paths = [
            [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
                ('w4', 'w&infin;', 'r')],
            [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
                ('w4', 'w&infin;', 'r')],
            [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w8', 'h'),
                ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w8', 'h'),
                ('w8', 'w&infin;', 'q')],
            [('w0', 'w5', 'j'), ('w5', 'w6', 'm'), ('w6', 'w7', 'o'),
                ('w7', 'w8', 'p'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w6', 'l'), ('w6', 'w7', 'o'), ('w7', 'w8', 'p'),
                ('w8', 'w&infin;', 'q')],
            [('w0', 'w3', 's'), ('w3', 'w4', 'g'), ('w4', 'w8', 'i'),
                ('w8', 'w&infin;', 'q')],
            [('w0', 'w3', 's'), ('w3', 'w4', 'g'), ('w4', 'w&infin;', 'r')],
            [('w0', 'w3', 's'), ('w3', 'w8', 'h'), ('w8', 'w&infin;', 'q')]
        ]
        for path in expected_paths:
            assert path in self.evidence.path_list
        assert len(expected_paths) == len(self.evidence.path_list)

        self.evidence.remove_vertex('w3', Evidence.UNCERTAIN)
        self.evidence.remove_vertex('w6', Evidence.UNCERTAIN)
        update_path_lists()
        self.evidence.add_edge('w2', 'w4', 'e', Evidence.UNCERTAIN)
        self.evidence.add_edge('w3', 'w4', 'g', Evidence.UNCERTAIN)
        self.evidence.add_edge('w3', 'w8', 'h', Evidence.UNCERTAIN)
        self.evidence.add_vertex('w1', Evidence.CERTAIN)

        reduced = self.evidence.reduced_graph
        fname = Util.create_path('out/Evidence_test_uncertain', False, 'pdf')
        reduced.create_figure(fname)

        expected_paths = [
            [('w0', 'w1', 'a'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
                ('w4', 'w&infin;', 'r')],
            [('w0', 'w1', 'a'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'b'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
                ('w4', 'w&infin;', 'r')],
            [('w0', 'w1', 'b'), ('w1', 'w2', 'c'), ('w2', 'w4', 'e'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w8', 'h'),
                ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'a'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
                ('w4', 'w&infin;', 'r')],
            [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w8', 'h'),
                ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
                ('w4', 'w8', 'i'), ('w8', 'w&infin;', 'q')],
            [('w0', 'w1', 'b'), ('w1', 'w3', 'd'), ('w3', 'w4', 'g'),
                ('w4', 'w&infin;', 'r')],
        ]
        for path in expected_paths:
            assert path in self.evidence.path_list
        assert len(expected_paths) == len(self.evidence.path_list)


class Test:
    def setup(self):
        self.node_prefix = 'w'
        self.sink_suffix = '&infin;'
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')

        self.st = StagedTree(
            dataframe=pd.read_excel(df_path),
            name="medical_staged"
        )
        self.st.calculate_AHC_transitions()
        self.ceg = ChainEventGraph(
            incoming_graph_data=self.st,
        )

    def test_creation_of_ceg(self) -> None:
        self.ceg.generate()
        fname = Util.create_path('out/medical_dm_CEG', True, 'pdf')
        self.ceg.create_figure(fname)
        assert os.path.isfile(fname)

    def test_generation(self) -> None:
        self.ceg.generate()
        path_list = self.ceg.path_list
        nodes = ['w0', 'w1', 'w4', 'w10', 'w9', 'w&infin;']

        subgraph = self.ceg.subgraph(nodes).copy()
        fname = Util.create_path('out/Subgraph_test', True, 'pdf')
        subgraph.create_figure(fname)
        print(path_list)
