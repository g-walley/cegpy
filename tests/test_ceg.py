import networkx as nx
from networkx.classes.function import nodes
import pandas as pd
from src.cegpy import StagedTree, Evidence, ChainEventGraph
from pathlib import Path
import pytest


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
        nx.relabel_nodes(self.ceg, {'s0': self.ceg.root_node}, copy=False)
        self.ceg._ChainEventGraph__trim_leaves_from_graph()
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
        nx.relabel_nodes(self.ceg, {'s0': self.ceg.root_node}, copy=False)
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

    def test_adding_evidence(self) -> None:
        certain_edges = [
            ('w1', 'w4', 'Experienced'),
            ('w4', 'w10', 'Hard'),
        ]
        for (u, v, k) in certain_edges:
            self.ceg.evidence.add_edge(u, v, k, Evidence.CERTAIN)
            assert (u, v, k) in self.ceg.evidence.certain_edges

        certain_vertices = {
            'w4', 'w10'
        }
        for vertex in certain_vertices:
            self.ceg.evidence.add_node(vertex, Evidence.CERTAIN)
            assert vertex in self.ceg.evidence.certain_nodes

        uncertain_edges = [
            ('w2', 'w5', 'Experienced'),
            ('w2', 'w6', 'Novice')
        ]
        for (u, v, k) in uncertain_edges:
            self.ceg.evidence.add_edge(u, v, k, Evidence.UNCERTAIN)
            assert (u, v, k) in self.ceg.evidence.uncertain_edges

        uncertain_vertices = {
            'w2', 'w13'
        }
        for vertex in uncertain_vertices:
            self.ceg.evidence.add_node(vertex, Evidence.UNCERTAIN)
            assert vertex in self.ceg.evidence.uncertain_vertices

    def test_propagation(self) -> None:
        self.ceg.generate()
        uncertain_edges = [
            ('w2', 'w5', 'Experienced'),
            ('w2', 'w6', 'Novice')
        ]
        certain_nodes = {
            'w12'
        }
        self.ceg.evidence.add_edges_from(uncertain_edges, Evidence.UNCERTAIN)
        self.ceg.evidence.add_nodes_from(certain_nodes, Evidence.CERTAIN)
        self.ceg.reduced

        self.ceg.clear_evidence()


class TestEvidence(object):
    def setup(self):
        G = nx.MultiDiGraph()
        nodes = ['w0', 'w1', 'w2', 'w3', 'w4',
                 'w5', 'w6', 'w7', 'w8', 'w&infin;']
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

    def test_add_and_remove_certain_edge(self):
        certain_edges = [
            ('w0', 'w1', 'a'),
            ('w1', 'w2', 'c'),
        ]
        for edge in certain_edges:
            self.evidence.add_certain_edge(*edge)
        assert certain_edges == self.evidence.certain_edges

        edge_to_remove = certain_edges[1]
        self.evidence.remove_certain_edge(*edge_to_remove)
        certain_edges.remove(edge_to_remove)
        assert certain_edges == self.evidence.certain_edges

        pytest.raises(
            ValueError,
            self.evidence.remove_certain_edge,
            *edge_to_remove,
        )

    def test_add_and_remove_certain_edge_list(self):
        certain_edges = [
            ('w0', 'w1', 'a'),
            ('w1', 'w2', 'c'),
            ('w2', 'w4', 'e'),
        ]
        self.evidence.add_certain_edge_list(certain_edges)
        assert certain_edges == self.evidence.certain_edges

        self.evidence.remove_certain_edge_list(certain_edges[1:])
        for edge in certain_edges[1:]:
            certain_edges.remove(edge)
        assert certain_edges == self.evidence.certain_edges

    def test_add_and_remove_uncertain_edge_set(self):
        uncertain_edge_sets = [
            {
                ('w0', 'w1', 'a'),
                ('w0', 'w1', 'b'),
                ('w0', 'w3', 's'),
            },
            {
                ('w0', 'w5', 'j'),
                ('w0', 'w7', 'k'),
            },
        ]
        for edge_set in uncertain_edge_sets:
            self.evidence.add_uncertain_edge_set(edge_set)
        assert uncertain_edge_sets == self.evidence.uncertain_edges

        edge_set_to_remove = uncertain_edge_sets[0].copy()
        self.evidence.remove_uncertain_edge_set(edge_set_to_remove)
        uncertain_edge_sets.remove(edge_set_to_remove)
        assert uncertain_edge_sets == self.evidence.uncertain_edges

        pytest.raises(
            ValueError,
            self.evidence.remove_uncertain_edge_set,
            edge_set_to_remove,
        )

    def test_add_and_remove_uncertain_edge_set_list(self):
        uncertain_edge_sets = [
            {
                ('w0', 'w1', 'a'),
                ('w0', 'w1', 'b'),
                ('w0', 'w3', 's'),
            },
            {
                ('w0', 'w5', 'j'),
                ('w0', 'w7', 'k'),
            },
            {
                ('w5', 'w6', 'm'),
                ('w5', 'w7', 'n'),
            }
        ]
        self.evidence.add_uncertain_edge_set_list(uncertain_edge_sets)
        assert uncertain_edge_sets == self.evidence.uncertain_edges

        self.evidence.remove_uncertain_edge_set_list(uncertain_edge_sets[1:])
        for edge_set in uncertain_edge_sets[1:]:
            uncertain_edge_sets.remove(edge_set)
        assert uncertain_edge_sets == self.evidence.uncertain_edges

    def test_add_and_remove_certain_nodes(self):
        # nodes = ['w0', 'w1', 'w2', 'w3', 'w4',
        #          'w5', 'w6', 'w7', 'w8', 'w&infin;']
        certain_nodes = {"w0", "w1", "w3"}
        for node in certain_nodes:
            self.evidence.add_certain_node(node)
        assert certain_nodes == self.evidence.certain_nodes

        pytest.raises(
            ValueError,
            self.evidence.add_certain_node,
            "w150",
        )

        node = certain_nodes.pop()
        self.evidence.remove_certain_node(node)
        assert certain_nodes == self.evidence.certain_nodes

        pytest.raises(
            KeyError,
            self.evidence.remove_certain_node,
            node
        )

    def test_add_and_remove_certain_nodes_set(self):
        certain_nodes = {"w0", "w1", "w3"}
        self.evidence.add_certain_node_set(certain_nodes)
        assert certain_nodes == self.evidence.certain_nodes

        nodes_to_remove = {"w0", "w1"}
        self.evidence.remove_certain_node_set(nodes_to_remove)
        certain_nodes.difference_update(nodes_to_remove)
        assert certain_nodes == self.evidence.certain_nodes

    def test_add_and_remove_uncertain_nodes_set(self):
        pass

    def test_add_and_remove_uncertain_nodes_set_list(self):
        pass

    def test_add_vertex_add_and_remove(self):
        uncertain_vertices = {'s1', 's2', 's3', 's45'}
        for vertex in uncertain_vertices:
            self.evidence.add_node(vertex, certain=False)
        assert uncertain_vertices == self.evidence.uncertain_vertices

        self.evidence.remove_node('s2', certain=False)
        uncertain_vertices.remove('s2')
        assert uncertain_vertices == self.evidence.uncertain_vertices

        certain_vertices = {'s1', 's2', 's3', 's45'}
        for vertex in certain_vertices:
            self.evidence.add_node(vertex, certain=True)
        assert certain_vertices == self.evidence.certain_nodes

        self.evidence.remove_node('s2', certain=True)
        certain_vertices.remove('s2')
        assert certain_vertices == self.evidence.certain_nodes
