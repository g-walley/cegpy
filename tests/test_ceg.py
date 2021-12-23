import networkx as nx
import pandas as pd
from src.cegpy import StagedTree, ChainEventGraph
from pathlib import Path


class TestUnitCEG(object):
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
        self.ceg = ChainEventGraph(self.st)

    def test_node_name_generation(self):
        prefix = self.ceg.node_prefix
        largest = 20
        node_names = [
            self.ceg._get_next_node_name()
            for _ in range(0, largest)
        ]
        assert (prefix + '1') == node_names[0]
        assert (prefix + str(largest)) == node_names[largest - 1]

    def test_trim_leaves_from_graph(self) -> None:
        self.ceg._trim_leaves_from_graph()
        for leaf in self.st.leaves:
            try:
                self.ceg.nodes[leaf]
                leaf_removed = False

            except KeyError:
                leaf_removed = True

            assert leaf_removed

            for edge_list_key in self.ceg.edges.keys():
                assert edge_list_key[1] != leaf

    def test_update_distances_of_nodes_to_sink(self) -> None:
        def check_distances():
            actual_node_dists = \
                nx.get_node_attributes(self.ceg, 'max_dist_to_sink')
            for node, distance in actual_node_dists.items():
                assert distance == expected_node_dists[node]
        root_node = self.node_prefix + '0'
        sink_node = self.node_prefix + self.sink_suffix
        expected_node_dists = {
            root_node: 4,
            's1': 3,
            's2': 3,
            's3': 2,
            's4': 2,
            's5': 2,
            's6': 2,
            's7': 2,
            's8': 2,
            's9': 1,
            's10': 1,
            's11': 1,
            's12': 1,
            's13': 1,
            's14': 1,
            's15': 1,
            's16': 1,
            's17': 1,
            's18': 1,
            's19': 1,
            's20': 1,
            sink_node: 0
        }
        nx.relabel_nodes(self.ceg, {'s0': self.ceg.root_node}, copy=False)
        self.ceg._trim_leaves_from_graph()
        self.ceg._update_distances_of_nodes_to_sink_node()
        check_distances()

        # Add another edge to the dictionary, to show that the path is max,
        # and not min distance to sink
        self.ceg.add_edge('s3', self.ceg.sink_node)
        self.ceg.add_edge('s1', self.ceg.sink_node)
        self.ceg.add_edge('s2', 's10')
        self.ceg._update_distances_of_nodes_to_sink_node()
        check_distances()

    def test_gen_nodes_with_increasing_distance(self) -> None:
        expected_nodes = {
            0: ['s21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
                's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38',
                's39', 's40', 's41', 's42', 's43', 's44'],
            1: ['s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                's18', 's19', 's20'],
            2: ['s3', 's4', 's5', 's6', 's7', 's8'],
            3: ['s1', 's2']
        }
        nx.relabel_nodes(self.ceg, {'s0': self.ceg.root_node}, copy=False)
        self.ceg._trim_leaves_from_graph()
        self.ceg._update_distances_of_nodes_to_sink_node()
        nodes_gen = self.ceg.\
            _gen_nodes_with_increasing_distance(
                start=0
            )

        for nodes in range(len(expected_nodes)):
            expected_node_list = expected_nodes[nodes]
            actual_node_list = next(nodes_gen)
            assert actual_node_list.sort() == expected_node_list.sort()
