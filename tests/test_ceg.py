# from os import name
from ..src.cegpy.graphs.ceg import ChainEventGraph
from ..src.cegpy.trees.staged import StagedTree
from ..src.cegpy.utilities.util import Util
# from collections import defaultdict
from pathlib import Path
import networkx as nx
import pandas as pd
import os.path

class TestCEG(object):
    def setup(self):
        self.node_prefix = 'w'
        self.sink_suffix = 'inf'
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
            node_prefix='w',
            sink_suffix='inf'
        )

    def test_node_name_generation(self):
        prefix = self.ceg.node_prefix
        largest = 20
        node_names = [
            self.ceg._ChainEventGraph__get_next_node_name()
            for _ in range(0, largest)
        ]
        assert (prefix + '1') == node_names[0]
        assert (prefix + str(largest)) == node_names[largest - 1]

    def test_trim_leaves_from_graph(self) -> None:
        self.ceg._ChainEventGraph__trim_leaves_from_graph()
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
        self.ceg._ChainEventGraph__update_distances_of_nodes_to_sink_node()
        check_distances()

        # Add another edge to the dictionary, to show that the path is max,
        # and not min distance to sink
        self.ceg.add_edge('s3', self.ceg.sink_node)
        self.ceg.add_edge('s1', self.ceg.sink_node)
        self.ceg.add_edge('s2', 's10')
        self.ceg._ChainEventGraph__update_distances_of_nodes_to_sink_node()
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
        self.ceg._ChainEventGraph__trim_leaves_from_graph()
        self.ceg._ChainEventGraph__update_distances_of_nodes_to_sink_node()
        nodes_gen = self.ceg.\
            _ChainEventGraph__gen_nodes_with_increasing_distance(
                start=0
            )

        for nodes in range(len(expected_nodes)):
            expected_node_list = expected_nodes[nodes]
            actual_node_list = next(nodes_gen)
            assert actual_node_list.sort() == expected_node_list.sort()

    def test_creation_of_ceg(self) -> None:
        self.ceg.generate_CEG()
        fname = Util.create_path('out/medical_dm_CEG', True, 'pdf')
        self.ceg.create_figure(fname)
        assert os.path.isfile(fname)

    def test_adding_evidence(self) -> None:
        self.ceg.generate_CEG()
        certain_variables = {
            self.st.get_variables()[1]: 'Experienced'
        }
        try:
            self.ceg.add_evidence(
                type_of_evidence='variables',
                evidence=certain_variables,
                certain=True
            )
            assert False  # We shouldn't be able to add variables!
        except ValueError:
            pass

        certain_edges = {
            ('s1', 's3'): 'Experienced',
            ('s3', 's12'): 'Hard'
        }
        self.ceg.add_evidence(
            type_of_evidence='edges',
            evidence=certain_edges,
            certain=True
        )

        certain_vertices = {
            's3', 's12'
        }
        self.ceg.add_evidence(
            type_of_evidence='vertices',
            evidence=certain_vertices,
            certain=True
        )

        uncertain_variables = {
            self.st.get_variables()[1]: ['Inexperienced', 'Novice'],
            self.st.get_variables()[2]: ['Easy']
        }
        try:
            self.ceg.add_evidence(
                type_of_evidence='variables',
                evidence=uncertain_variables,
                certain=False
            )
            assert False  # We shouldn't be able to add variables!
        except ValueError:
            pass

        uncertain_edges = {
            ('s2', 's8'): 'Experienced',
            ('s6', 's15'): 'Hard'
        }
        self.ceg.add_evidence(
            type_of_evidence='edges',
            evidence=uncertain_edges,
            certain=False
        )

        uncertain_vertices = {
            's2', 's15'
        }
        self.ceg.add_evidence(
            type_of_evidence='vertices',
            evidence=uncertain_vertices,
            certain=False
        )

        print(self.ceg.get_evidence_str())

        self.ceg.clear_evidence()
        evidence = self.ceg.get_evidence_dict()
        assert evidence['certain'] == {
            'edges': {
                'evidence': {},
                'paths': set()
            },
            'vertices': {
                'evidence': set(),
                'paths': set()
            }
        }
        assert evidence['uncertain'] == {
            'edges': {
                'evidence': {},
                'paths': set()
            },
            'vertices': {
                'evidence': set(),
                'paths': set()
            }
        }

    def test_find_paths_from_edge(self) -> None:
        self.ceg.generate_CEG()
        expected_paths = [
            [
                ('s0', 's1', 'Blast'),
                ('s1', 's3', 'Experienced'),
                ('s3', 's9', 'Easy'),
                ('s9', 'w_inf', 'Blast')
            ],
            [
                ('s0', 's1', 'Blast'),
                ('s1', 's3', 'Experienced'),
                ('s3', 's9', 'Easy'),
                ('s9', 'w_inf', 'Non-blast')
            ],
            [
                ('s0', 's1', 'Blast'),
                ('s1', 's3', 'Inexperienced'),
                ('s3', 's9', 'Easy'),
                ('s9', 'w_inf', 'Blast')
            ],
            [
                ('s0', 's1', 'Blast'),
                ('s1', 's3', 'Inexperienced'),
                ('s3', 's9', 'Easy'),
                ('s9', 'w_inf', 'Non-blast')
            ]
        ]
        expected_path_set = set(map(frozenset, expected_paths))
        edge = ('s3', 's9', 'Easy')
        actual_path_set = self.ceg._find_paths_containing_edge(edge)
        assert expected_path_set == actual_path_set

    def test_find_paths_from_node(self) -> None:
        self.ceg.generate_CEG()
        sink_actual_paths = self.ceg._find_paths_containing_node('w_inf')
        assert len(sink_actual_paths) == 24
        root_actual_paths = self.ceg._find_paths_containing_node('s0')
        assert len(root_actual_paths) == 24

        expected_paths = [
            [
                ('s0', 's1', 'Blast'),
                ('s1', 's3', 'Experienced'),
                ('s3', 's9', 'Easy'),
                ('s9', 'w_inf', 'Blast')
            ],
            [
                ('s0', 's1', 'Blast'),
                ('s1', 's3', 'Experienced'),
                ('s3', 's9', 'Easy'),
                ('s9', 'w_inf', 'Non-blast')
            ],
            [
                ('s0', 's1', 'Blast'),
                ('s1', 's3', 'Inexperienced'),
                ('s3', 's9', 'Easy'),
                ('s9', 'w_inf', 'Blast')
            ],
            [
                ('s0', 's1', 'Blast'),
                ('s1', 's3', 'Inexperienced'),
                ('s3', 's9', 'Easy'),
                ('s9', 'w_inf', 'Non-blast')
            ]
        ]
        expected_path_set = set(map(frozenset, expected_paths))
        s_nine_actual_path_set = self.ceg._find_paths_containing_node('s9')

        assert s_nine_actual_path_set == expected_path_set
