from ..src.cegpy.graphs.ceg import ChainEventGraph
from ..src.cegpy.trees.staged import StagedTree
# from collections import defaultdict
from pathlib import Path
import pandas as pd


class TestCEG(object):
    def setup(self):
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        st_params = {
            'dataframe': pd.read_excel(df_path)
        }
        self.st = StagedTree(st_params)
        self.st.calculate_AHC_transitions()
        self.ceg = ChainEventGraph(
            staged_tree=self.st,
            root='s0'
        )

    def test_integration(self):

        self.ceg._create_graph_representation()

    def test_create_graph_representation(self) -> None:
        nodes = self.st.get_nodes()
        edges = self.st.get_edges()
        edge_labels = self.st.get_edge_labels()

        # check that all nodes exist
        for node in nodes:
            assert self.ceg.graph['nodes'].get(node) is not None

        # check edges were created correctly
        for idx, edge in enumerate(edges):
            edge_list = self.ceg.graph['edges'].get(edge)
            src, dest = edge
            assert edge_list is not None
            label_found = False
            for elem in edge_list:
                label_found = True \
                    if elem['label'] == edge_labels[idx][-1] \
                    else False
                assert src == elem['src']
                assert dest == elem['dest']
            assert label_found

    def test_trim_leaves_from_graph(self) -> None:
        new_graph, _ = self.ceg._trim_leaves_from_graph(self.ceg.graph)
        leaves = self.st.get_leaves()
        for leaf in leaves:
            try:
                new_graph['nodes'][leaf]
                leaf_removed = False

            except KeyError:
                leaf_removed = True

            assert leaf_removed

            for edge_list_key in new_graph['edges'].keys():
                assert edge_list_key[1] != leaf

    def test_identify_root_node(self) -> None:
        root = self.ceg._identify_root_node(self.ceg.graph)
        assert root == 's0'

    def test_merge_nodes(self) -> None:
        new_graph = self.ceg._merge_nodes()

    def test_initialisation_of_ceg(self) -> None:
        pass
