import networkx as nx
import pytest
from src.cegpy.graphs import (
    ChainEventGraph,
    TransporterChainEventGraph
)


class TestTransporterCEG(object):
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
        H._update_path_list()
        self.tceg = TransporterChainEventGraph(H)

    def test_repr(self):
        rep = repr(self.tceg)

        assert "certain_edges=" in rep
        assert "certain_nodes=" in rep
        assert "uncertain_edges=" in rep
        assert "uncertain_nodes=" in rep

    def test_update_path_list(self):
        pre_path_list = self.tceg.paths

        certain_edges = [
            ('w0', 'w1', 'a'),
            ('w1', 'w2', 'c'),
        ]
        for edge in certain_edges:
            self.tceg.add_certain_edge(*edge)

        self.tceg._update_path_list()
        after_path_list = self.tceg.paths
        assert pre_path_list != after_path_list

    def test_add_and_remove_certain_edge(self):
        certain_edges = [
            ('w0', 'w1', 'a'),
            ('w1', 'w2', 'c'),
        ]
        for edge in certain_edges:
            self.tceg.add_certain_edge(*edge)
        assert certain_edges == self.tceg.certain_edges

        pytest.raises(
            ValueError,
            self.tceg.add_certain_edge,
            *('w3', 'w100', 'h')
        )

        edge_to_remove = certain_edges[1]
        self.tceg.remove_certain_edge(*edge_to_remove)
        certain_edges.remove(edge_to_remove)
        assert certain_edges == self.tceg.certain_edges

        pytest.raises(
            ValueError,
            self.tceg.remove_certain_edge,
            *edge_to_remove,
        )

    def test_add_and_remove_certain_edge_list(self):
        certain_edges = [
            ('w0', 'w1', 'a'),
            ('w1', 'w2', 'c'),
            ('w2', 'w4', 'e'),
        ]
        self.tceg.add_certain_edge_list(certain_edges)
        assert certain_edges == self.tceg.certain_edges

        self.tceg.remove_certain_edge_list(certain_edges[1:])
        for edge in certain_edges[1:]:
            certain_edges.remove(edge)
        assert certain_edges == self.tceg.certain_edges

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
            self.tceg.add_uncertain_edge_set(edge_set)
        assert uncertain_edge_sets == self.tceg.uncertain_edges

        pytest.raises(
            ValueError,
            self.tceg.add_uncertain_edge_set,
            {
                ('w0', 'w5', 'j'),
                ('w0', 'w7', 'k'),
                ('w3', 'w100', 'h')
            },
        )

        edge_set_to_remove = uncertain_edge_sets[0].copy()
        self.tceg.remove_uncertain_edge_set(edge_set_to_remove)
        uncertain_edge_sets.remove(edge_set_to_remove)
        assert uncertain_edge_sets == self.tceg.uncertain_edges

        pytest.raises(
            ValueError,
            self.tceg.remove_uncertain_edge_set,
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
        self.tceg.add_uncertain_edge_set_list(uncertain_edge_sets)
        assert uncertain_edge_sets == self.tceg.uncertain_edges

        self.tceg.remove_uncertain_edge_set_list(uncertain_edge_sets[1:])
        for edge_set in uncertain_edge_sets[1:]:
            uncertain_edge_sets.remove(edge_set)
        assert uncertain_edge_sets == self.tceg.uncertain_edges

    def test_add_and_remove_certain_nodes(self):
        # nodes = ['w0', 'w1', 'w2', 'w3', 'w4',
        #          'w5', 'w6', 'w7', 'w8', 'w&infin;']
        certain_nodes = {"w0", "w1", "w3"}
        for node in certain_nodes:
            self.tceg.add_certain_node(node)
        assert certain_nodes == self.tceg.certain_nodes

        pytest.raises(
            ValueError,
            self.tceg.add_certain_node,
            "w150",
        )

        node = certain_nodes.pop()
        self.tceg.remove_certain_node(node)
        assert certain_nodes == self.tceg.certain_nodes

        pytest.raises(
            ValueError,
            self.tceg.remove_certain_node,
            node
        )

    def test_add_and_remove_certain_nodes_set(self):
        certain_nodes = {"w0", "w1", "w3"}
        self.tceg.add_certain_node_set(certain_nodes)
        assert certain_nodes == self.tceg.certain_nodes

        nodes_to_remove = {"w0", "w1"}
        self.tceg.remove_certain_node_set(nodes_to_remove)
        certain_nodes.difference_update(nodes_to_remove)
        assert certain_nodes == self.tceg.certain_nodes

    def test_add_and_remove_uncertain_nodes_set(self):
        uncertain_node_sets = [
            {"w0", "w1", "w3"},
            {"w4", "w5"},
            {"w7", "w8"},
        ]
        for node_set in uncertain_node_sets:
            self.tceg.add_uncertain_node_set(node_set)
        assert uncertain_node_sets == self.tceg.uncertain_nodes

        pytest.raises(
            ValueError,
            self.tceg.add_uncertain_node_set,
            {"w6", "w150"},
        )
        node_set = uncertain_node_sets.pop()
        self.tceg.remove_uncertain_node_set(node_set)
        assert uncertain_node_sets == self.tceg.uncertain_nodes

        pytest.raises(
            ValueError,
            self.tceg.remove_uncertain_node_set,
            node_set
        )

    def test_add_and_remove_uncertain_nodes_set_list(self):
        uncertain_node_sets = [
            {"w0", "w1", "w3"},
            {"w4", "w5"},
            {"w7", "w8"},
        ]
        self.tceg.add_uncertain_node_set_list(uncertain_node_sets)
        assert uncertain_node_sets == self.tceg.uncertain_nodes

        node_sets_to_remove = uncertain_node_sets[1:]
        self.tceg.remove_uncertain_node_set_list(node_sets_to_remove)
        for node_set in node_sets_to_remove:
            uncertain_node_sets.remove(node_set)
        assert uncertain_node_sets == self.tceg.uncertain_nodes

    def test_clear_all_evidence(self):
        self.tceg.add_certain_edge("w0", "w1", "a")
        self.tceg.add_certain_node("w1")
        self.tceg.add_uncertain_edge_set({
            ('w0', 'w1', 'a'),
            ('w0', 'w1', 'b'),
        })
        self.tceg.add_uncertain_node_set({"w1", "w2"})

        self.tceg.clear_all_evidence()
        assert self.tceg.certain_edges == []
        assert self.tceg.certain_nodes == set()
        assert self.tceg.uncertain_edges == []
        assert self.tceg.uncertain_nodes == []

    def _test_propagation(self) -> None:

        self.tceg._ceg.generate()
        uncertain_edges = {
            ('w2', 'w5', 'Experienced'),
            ('w2', 'w6', 'Novice')
        }
        certain_nodes = {
            'w12'
        }
        self.ceg.evidence.add_uncertain_edge_set(uncertain_edges)
        self.ceg.evidence.add_certain_node_set(certain_nodes)
        self.ceg.reduced

        self.ceg.clear_evidence()
