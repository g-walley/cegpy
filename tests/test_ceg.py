from ..src.cegpy.graphs.ceg import ChainEventGraph
from ..src.cegpy.trees.staged import StagedTree
from ..src.cegpy.utilities.util import Util
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
        _ = self.ceg._trim_leaves_from_graph(self.ceg.graph)
        leaves = self.st.get_leaves()
        for leaf in leaves:
            try:
                self.ceg.graph['nodes'][leaf]
                leaf_removed = False

            except KeyError:
                leaf_removed = True

            assert leaf_removed

            for edge_list_key in self.ceg.graph['edges'].keys():
                assert edge_list_key[1] != leaf

    def test_identify_root_node(self) -> None:
        root = self.ceg._identify_root_node(self.ceg.graph)
        assert root == 's0'

    def test_update_distances_of_nodes_to_sink(self) -> None:
        def create_node_dist_dict(graph) -> dict:
            node_dists = {}
            for key in graph['nodes'].keys():
                node_dists[key] = graph['nodes'][key]['max_dist_to_sink']
            return node_dists

        med_graph = self.ceg.graph.copy()
        expected_node_dists = {
            's0': 4,
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
            's21': 0,
            's22': 0,
            's23': 0,
            's24': 0,
            's25': 0,
            's26': 0,
            's27': 0,
            's28': 0,
            's29': 0,
            's30': 0,
            's31': 0,
            's32': 0,
            's33': 0,
            's34': 0,
            's35': 0,
            's36': 0,
            's37': 0,
            's38': 0,
            's39': 0,
            's40': 0,
            's41': 0,
            's42': 0,
            's43': 0,
            's44': 0
        }
        self.ceg._update_distances_of_nodes_to_sink(
            med_graph, self.st.get_leaves().copy()
        )
        actual_node_dists = create_node_dist_dict(med_graph)
        assert actual_node_dists == expected_node_dists

        new_edge_1 = ('s3', 's24')
        new_edge_2 = ('s1', 's29')
        # Add another edge to the dictionary, to show that the path is max,
        # and not min distance to sink
        med_graph['edges'][new_edge_1] = [
            self.ceg._create_new_edge(
                src=new_edge_1[0], dest=new_edge_2[1], label='new_edge_1'
            )
        ]
        med_graph['nodes'][new_edge_1[1]]['ingoing_edges'].append(new_edge_1)
        med_graph['nodes'][new_edge_1[0]]['outgoing_edges'].append(new_edge_1)

        # Add a second edge to the dictionary
        med_graph['edges'][new_edge_2] = [
            self.ceg._create_new_edge(
                src=new_edge_2[0], dest=new_edge_2[1], label='new_edge_2'
            )
        ]
        med_graph['nodes'][new_edge_2[1]]['ingoing_edges'].append(new_edge_2)
        med_graph['nodes'][new_edge_2[0]]['outgoing_edges'].append(new_edge_2)

        self.ceg._update_distances_of_nodes_to_sink(
            med_graph, self.st.get_leaves().copy()
        )
        actual_node_dists = create_node_dist_dict(med_graph)
        assert actual_node_dists == expected_node_dists

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

        nodes_gen = self.ceg._gen_nodes_with_increasing_distance(
            graph=self.ceg.graph,
            start=0
        )

        for nodes in range(len(expected_nodes)):
            expected_node_list = expected_nodes[nodes]
            actual_node_list = next(nodes_gen)
            assert actual_node_list.sort() == expected_node_list.sort()

    def test_creation_of_ceg(self) -> None:
        self.ceg.generate_CEG()
        # path = Util.create_path('out/medical_dm_CEG', True, 'pdf')
        # string = self.ceg.get_evidence_str()
        # self.ceg.create_figure(path)
        pass

    def test_adding_evidence(self) -> None:
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
