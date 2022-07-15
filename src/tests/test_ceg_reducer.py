import unittest
import networkx as nx
import pytest
from pydotplus.graphviz import InvocationException
from cegpy import ChainEventGraph, ChainEventGraphReducer


class TestCEGReducer(unittest.TestCase):
    def setUp(self):
        G = nx.MultiDiGraph()
        self.init_nodes = [
            "w0",
            "w1",
            "w2",
            "w3",
            "w4",
            "w5",
            "w6",
            "w7",
            "w8",
            "w_infinity",
        ]
        self.init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w1", "b"),
            ("w1", "w2", "c"),
            ("w1", "w3", "d"),
            ("w2", "w4", "e"),
            ("w2", "w4", "f"),
            ("w3", "w4", "g"),
            ("w3", "w8", "h"),
            ("w4", "w8", "i"),
            ("w0", "w5", "j"),
            ("w0", "w7", "k"),
            ("w0", "w6", "l"),
            ("w5", "w6", "m"),
            ("w5", "w7", "n"),
            ("w6", "w7", "o"),
            ("w7", "w8", "p"),
            ("w8", "w_infinity", "q"),
            ("w4", "w_infinity", "r"),
            ("w0", "w3", "s"),
        ]
        G.add_nodes_from(self.init_nodes)
        G.add_edges_from(self.init_edges)
        G.root = "w0"
        G.ahc_output = {
            "Merged Situations": [("s1", "s2")],
            "Loglikelihood": 1234.5678,
        }
        H = ChainEventGraph(G, generate=False)
        self.rceg = ChainEventGraphReducer(H)

        self.rceg._ceg.edges["w0", "w1", "a"]["probability"] = 0.3
        self.rceg._ceg.edges["w0", "w1", "b"]["probability"] = 0.3
        self.rceg._ceg.edges["w1", "w2", "c"]["probability"] = 0.88
        self.rceg._ceg.edges["w1", "w3", "d"]["probability"] = 0.12
        self.rceg._ceg.edges["w2", "w4", "e"]["probability"] = 0.95
        self.rceg._ceg.edges["w2", "w4", "f"]["probability"] = 0.05
        self.rceg._ceg.edges["w3", "w4", "g"]["probability"] = 0.5
        self.rceg._ceg.edges["w3", "w8", "h"]["probability"] = 0.5
        self.rceg._ceg.edges["w4", "w8", "i"]["probability"] = 0.9
        self.rceg._ceg.edges["w0", "w5", "j"]["probability"] = 0.2
        self.rceg._ceg.edges["w0", "w7", "k"]["probability"] = 0.05
        self.rceg._ceg.edges["w0", "w6", "l"]["probability"] = 0.05
        self.rceg._ceg.edges["w5", "w6", "m"]["probability"] = 0.6
        self.rceg._ceg.edges["w5", "w7", "n"]["probability"] = 0.4
        self.rceg._ceg.edges["w6", "w7", "o"]["probability"] = 1
        self.rceg._ceg.edges["w7", "w8", "p"]["probability"] = 1
        self.rceg._ceg.edges["w8", "w_infinity", "q"]["probability"] = 1
        self.rceg._ceg.edges["w4", "w_infinity", "r"]["probability"] = 0.1
        self.rceg._ceg.edges["w0", "w3", "s"]["probability"] = 0.1

        try:
            self.rceg._ceg.create_figure("out/test_propation_pre.pdf")
        except InvocationException:
            pass

    def test_repr(self):
        rep = repr(self.rceg)

        assert "certain_edges=" in rep
        assert "certain_nodes=" in rep
        assert "uncertain_edges=" in rep
        assert "uncertain_nodes=" in rep

    def test_update_path_list(self):
        pre_path_list = self.rceg.paths

        certain_edges = [
            ("w0", "w1", "a"),
            ("w1", "w2", "c"),
        ]
        for edge in certain_edges:
            self.rceg.add_certain_edge(*edge)

        self.rceg._update_path_list()
        after_path_list = self.rceg.paths
        assert pre_path_list != after_path_list

    def test_add_and_remove_certain_edge(self):
        certain_edges = [
            ("w0", "w1", "a"),
            ("w1", "w2", "c"),
        ]
        for edge in certain_edges:
            self.rceg.add_certain_edge(*edge)
        assert certain_edges == self.rceg.certain_edges

        pytest.raises(ValueError, self.rceg.add_certain_edge, *("w3", "w100", "h"))

        edge_to_remove = certain_edges[1]
        self.rceg.remove_certain_edge(*edge_to_remove)
        certain_edges.remove(edge_to_remove)
        assert certain_edges == self.rceg.certain_edges

        pytest.raises(
            ValueError,
            self.rceg.remove_certain_edge,
            *edge_to_remove,
        )

    def test_add_and_remove_certain_edge_list(self):
        certain_edges = [
            ("w0", "w1", "a"),
            ("w1", "w2", "c"),
            ("w2", "w4", "e"),
        ]
        self.rceg.add_certain_edge_list(certain_edges)
        assert certain_edges == self.rceg.certain_edges

        self.rceg.remove_certain_edge_list(certain_edges[1:])
        for edge in certain_edges[1:]:
            certain_edges.remove(edge)
        assert certain_edges == self.rceg.certain_edges

    def test_add_and_remove_uncertain_edge_set(self):
        uncertain_edge_sets = [
            {
                ("w0", "w1", "a"),
                ("w0", "w1", "b"),
                ("w0", "w3", "s"),
            },
            {
                ("w0", "w5", "j"),
                ("w0", "w7", "k"),
            },
        ]
        for edge_set in uncertain_edge_sets:
            self.rceg.add_uncertain_edge_set(edge_set)
        assert uncertain_edge_sets == self.rceg.uncertain_edges

        pytest.raises(
            ValueError,
            self.rceg.add_uncertain_edge_set,
            {("w0", "w5", "j"), ("w0", "w7", "k"), ("w3", "w100", "h")},
        )

        edge_set_to_remove = uncertain_edge_sets[0].copy()
        self.rceg.remove_uncertain_edge_set(edge_set_to_remove)
        uncertain_edge_sets.remove(edge_set_to_remove)
        assert uncertain_edge_sets == self.rceg.uncertain_edges

        pytest.raises(
            ValueError,
            self.rceg.remove_uncertain_edge_set,
            edge_set_to_remove,
        )

    def test_add_and_remove_uncertain_edge_set_list(self):
        uncertain_edge_sets = [
            {
                ("w0", "w1", "a"),
                ("w0", "w1", "b"),
                ("w0", "w3", "s"),
            },
            {
                ("w0", "w5", "j"),
                ("w0", "w7", "k"),
            },
            {
                ("w5", "w6", "m"),
                ("w5", "w7", "n"),
            },
        ]
        self.rceg.add_uncertain_edge_set_list(uncertain_edge_sets)
        assert uncertain_edge_sets == self.rceg.uncertain_edges

        self.rceg.remove_uncertain_edge_set_list(uncertain_edge_sets[1:])
        for edge_set in uncertain_edge_sets[1:]:
            uncertain_edge_sets.remove(edge_set)
        assert uncertain_edge_sets == self.rceg.uncertain_edges

    def test_add_and_remove_certain_nodes(self):
        # nodes = ['w0', 'w1', 'w2', 'w3', 'w4',
        #          'w5', 'w6', 'w7', 'w8', 'w_infinity']
        certain_nodes = {"w0", "w1", "w3"}
        for node in certain_nodes:
            self.rceg.add_certain_node(node)
        assert certain_nodes == self.rceg.certain_nodes

        pytest.raises(
            ValueError,
            self.rceg.add_certain_node,
            "w150",
        )

        node = certain_nodes.pop()
        self.rceg.remove_certain_node(node)
        assert certain_nodes == self.rceg.certain_nodes

        pytest.raises(ValueError, self.rceg.remove_certain_node, node)

    def test_add_and_remove_certain_nodes_set(self):
        certain_nodes = {"w0", "w1", "w3"}
        self.rceg.add_certain_node_set(certain_nodes)
        assert certain_nodes == self.rceg.certain_nodes

        nodes_to_remove = {"w0", "w1"}
        self.rceg.remove_certain_node_set(nodes_to_remove)
        certain_nodes.difference_update(nodes_to_remove)
        assert certain_nodes == self.rceg.certain_nodes

    def test_add_and_remove_uncertain_nodes_set(self):
        uncertain_node_sets = [
            {"w0", "w1", "w3"},
            {"w4", "w5"},
            {"w7", "w8"},
        ]
        for node_set in uncertain_node_sets:
            self.rceg.add_uncertain_node_set(node_set)
        assert uncertain_node_sets == self.rceg.uncertain_nodes

        pytest.raises(
            ValueError,
            self.rceg.add_uncertain_node_set,
            {"w6", "w150"},
        )
        node_set = uncertain_node_sets.pop()
        self.rceg.remove_uncertain_node_set(node_set)
        assert uncertain_node_sets == self.rceg.uncertain_nodes

        pytest.raises(ValueError, self.rceg.remove_uncertain_node_set, node_set)

    def test_add_and_remove_uncertain_nodes_set_list(self):
        uncertain_node_sets = [
            {"w0", "w1", "w3"},
            {"w4", "w5"},
            {"w7", "w8"},
        ]
        self.rceg.add_uncertain_node_set_list(uncertain_node_sets)
        assert uncertain_node_sets == self.rceg.uncertain_nodes

        node_sets_to_remove = uncertain_node_sets[1:]
        self.rceg.remove_uncertain_node_set_list(node_sets_to_remove)
        for node_set in node_sets_to_remove:
            uncertain_node_sets.remove(node_set)
        assert uncertain_node_sets == self.rceg.uncertain_nodes

    def test_clear_all_evidence(self):
        self.rceg.add_certain_edge("w0", "w1", "a")
        self.rceg.add_certain_node("w1")
        self.rceg.add_uncertain_edge_set(
            {
                ("w0", "w1", "a"),
                ("w0", "w1", "b"),
            }
        )
        self.rceg.add_uncertain_node_set({"w1", "w2"})

        self.rceg.clear_all_evidence()
        assert self.rceg.certain_edges == []
        assert self.rceg.certain_nodes == set()
        assert self.rceg.uncertain_edges == []
        assert self.rceg.uncertain_nodes == []

    def test_propagation(self) -> None:
        uncertain_edges = {
            ("w3", "w4", "g"),
            ("w3", "w8", "h"),
        }
        certain_nodes = {"w1"}
        self.rceg.add_uncertain_edge_set(uncertain_edges)
        self.rceg.add_certain_node_set(certain_nodes)
        rceg_out = self.rceg.graph

        try:
            rceg_out.create_figure("out/test_propagation_after.pdf")
        except InvocationException:
            pass

    def test_propagation_two(self) -> None:
        uncertain_edges = {
            ("w2", "w4", "e"),
            ("w3", "w4", "g"),
            ("w3", "w8", "h"),
        }
        self.rceg.add_uncertain_edge_set(uncertain_edges)
        self.rceg.add_certain_node("w1")
        rceg_out = self.rceg.graph
        assert rceg_out.edges["w0", "w1", "a"]["probability"] == 0.5
        assert rceg_out.edges["w0", "w1", "b"]["probability"] == 0.5
        assert rceg_out.edges["w1", "w2", "c"]["probability"] == (
            pytest.approx(0.87, abs=0.01)
        )
        assert rceg_out.edges["w1", "w3", "d"]["probability"] == (
            pytest.approx(0.13, abs=0.01)
        )
        assert rceg_out.edges["w2", "w4", "e"]["probability"] == 1.0
        assert rceg_out.edges["w3", "w4", "g"]["probability"] == 0.5
        assert rceg_out.edges["w3", "w8", "h"]["probability"] == 0.5
        assert rceg_out.edges["w4", "w8", "i"]["probability"] == 0.9
        assert rceg_out.edges["w8", "w_infinity", "q"]["probability"] == 1.0
        assert rceg_out.edges["w4", "w_infinity", "r"]["probability"] == 0.1
        try:
            rceg_out.create_figure("out/test_propagation_two.pdf")
        except InvocationException:
            pass

    def test_propagation_three(self) -> None:
        uncertain_nodes = {"w3", "w6"}
        self.rceg.add_uncertain_node_set(uncertain_nodes)
        rceg_out = self.rceg.graph
        try:
            rceg_out.create_figure("out/test_propagation_three.pdf")
        except InvocationException:
            pass
        assert rceg_out.edges["w0", "w1", "a"]["probability"] == (
            pytest.approx(0.11, abs=0.01)
        )
        assert rceg_out.edges["w0", "w1", "b"]["probability"] == (
            pytest.approx(0.11, abs=0.01)
        )
        assert rceg_out.edges["w1", "w3", "d"]["probability"] == (
            pytest.approx(1.0, abs=0.01)
        )
        assert rceg_out.edges["w3", "w4", "g"]["probability"] == (
            pytest.approx(0.5, abs=0.01)
        )
        assert rceg_out.edges["w3", "w8", "h"]["probability"] == (
            pytest.approx(0.5, abs=0.01)
        )
        assert rceg_out.edges["w4", "w8", "i"]["probability"] == (
            pytest.approx(0.9, abs=0.01)
        )
        assert rceg_out.edges["w0", "w5", "j"]["probability"] == (
            pytest.approx(0.35, abs=0.01)
        )
        assert rceg_out.edges["w0", "w6", "l"]["probability"] == (
            pytest.approx(0.15, abs=0.01)
        )
        assert rceg_out.edges["w5", "w6", "m"]["probability"] == (
            pytest.approx(1.0, abs=0.01)
        )
        assert rceg_out.edges["w6", "w7", "o"]["probability"] == (
            pytest.approx(1.0, abs=0.01)
        )
        assert rceg_out.edges["w7", "w8", "p"]["probability"] == (
            pytest.approx(1.0, abs=0.01)
        )
        assert rceg_out.edges["w8", "w_infinity", "q"]["probability"] == (
            pytest.approx(1.0, abs=0.01)
        )
        assert rceg_out.edges["w4", "w_infinity", "r"]["probability"] == (
            pytest.approx(0.1, abs=0.01)
        )
        assert rceg_out.edges["w0", "w3", "s"]["probability"] == (
            pytest.approx(0.29, abs=0.01)
        )

    def test_propagation_four(self) -> None:
        uncertain_edges = [
            {
                ("w2", "w4", "e"),
                ("w3", "w4", "g"),
                ("w3", "w8", "h"),
            },
            {
                ("w1", "w2", "c"),
                ("w1", "w3", "d"),
            },
        ]
        self.rceg.add_uncertain_edge_set_list(uncertain_edges)

        rceg_out = self.rceg.graph
        assert rceg_out.edges["w0", "w1", "a"]["probability"] == 0.5
        assert rceg_out.edges["w0", "w1", "b"]["probability"] == 0.5
        assert rceg_out.edges["w1", "w2", "c"]["probability"] == (
            pytest.approx(0.87, abs=0.01)
        )
        assert rceg_out.edges["w1", "w3", "d"]["probability"] == (
            pytest.approx(0.13, abs=0.01)
        )
        assert rceg_out.edges["w2", "w4", "e"]["probability"] == 1.0
        assert rceg_out.edges["w3", "w4", "g"]["probability"] == 0.5
        assert rceg_out.edges["w3", "w8", "h"]["probability"] == 0.5
        assert rceg_out.edges["w4", "w8", "i"]["probability"] == 0.9
        assert rceg_out.edges["w8", "w_infinity", "q"]["probability"] == 1.0
        assert rceg_out.edges["w4", "w_infinity", "r"]["probability"] == 0.1
        try:
            rceg_out.create_figure("out/test_propagation_four.pdf")
        except InvocationException:
            pass

    def test_str_out(self):
        uncertain_edges = {
            ("w7", "w8", "p"),
            ("w3", "w8", "h"),
        }
        self.rceg.add_uncertain_edge_set(uncertain_edges)

        str_rep = self.rceg.__str__()
        for edge in uncertain_edges:
            assert str(edge) in str_rep

        certain_edges = [
            ("w0", "w5", "j"),
            ("w5", "w7", "n"),
        ]
        self.rceg.add_certain_edge_list(certain_edges)
        str_rep = self.rceg.__str__()
        for edge in certain_edges:
            assert str(edge) in str_rep

        uncertain_nodes = [
            {"w5", "w6"},
            {"w3", "w4"},
        ]
        self.rceg.add_uncertain_node_set_list(uncertain_nodes)
        str_rep = self.rceg.__str__()
        for node_set in uncertain_nodes:
            assert str(node_set) in str_rep

        certain_nodes = {"w1", "w2"}
        self.rceg.add_certain_node_set(certain_nodes)

        str_rep = self.rceg.__str__()
        for edge in uncertain_edges:
            assert str(edge) in str_rep

        for edge in certain_edges:
            assert str(edge) in str_rep

        for node_set in uncertain_nodes:
            assert str(node_set) in str_rep

        for node in certain_nodes:
            assert str(node) in str_rep


class TestReducedCEGTwo(object):
    def setup(self):
        G = nx.MultiDiGraph()
        self.init_nodes = ["w0", "w1", "w2", "w3", "w4", "w5", "w6", "w_infinity"]
        self.init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w0", "w3", "c"),
            ("w1", "w4", "d"),
            ("w1", "w4", "e"),
            ("w2", "w4", "f"),
            ("w2", "w5", "g"),
            ("w2", "w6", "h"),
            ("w3", "w6", "i"),
            ("w4", "w_infinity", "j"),
            ("w4", "w5", "k"),
            ("w5", "w_infinity", "l"),
            ("w6", "w_infinity", "m"),
            ("w2", "w3", "n"),
            ("w5", "w6", "o"),
        ]
        G.add_nodes_from(self.init_nodes)
        G.add_edges_from(self.init_edges)
        G.root = "w0"
        G.ahc_output = {
            "Merged Situations": [("s1", "s2")],
            "Loglikelihood": 1234.5678,
        }
        H = ChainEventGraph(G, generate=False)
        self.rceg = ChainEventGraphReducer(H)
        self.rceg._ceg.edges["w0", "w1", "a"]["probability"] = 0.3
        self.rceg._ceg.edges["w0", "w2", "b"]["probability"] = 0.4
        self.rceg._ceg.edges["w0", "w3", "c"]["probability"] = 0.3
        self.rceg._ceg.edges["w1", "w4", "d"]["probability"] = 0.35
        self.rceg._ceg.edges["w1", "w4", "e"]["probability"] = 0.65
        self.rceg._ceg.edges["w2", "w4", "f"]["probability"] = 0.25
        self.rceg._ceg.edges["w2", "w5", "g"]["probability"] = 0.5
        self.rceg._ceg.edges["w2", "w6", "h"]["probability"] = 0.15
        self.rceg._ceg.edges["w3", "w6", "i"]["probability"] = 1.0
        self.rceg._ceg.edges["w4", "w_infinity", "j"]["probability"] = 0.8
        self.rceg._ceg.edges["w4", "w5", "k"]["probability"] = 0.2
        self.rceg._ceg.edges["w5", "w_infinity", "l"]["probability"] = 0.7
        self.rceg._ceg.edges["w6", "w_infinity", "m"]["probability"] = 1.0
        self.rceg._ceg.edges["w2", "w3", "n"]["probability"] = 0.1
        self.rceg._ceg.edges["w5", "w6", "o"]["probability"] = 0.3
        try:
            self.rceg._ceg.create_figure("out/test_propation_pre.pdf")
        except InvocationException:
            pass

    def test_propagation_one(self):
        uncertain_nodes = {"w4", "w5"}
        self.rceg.add_uncertain_node_set(uncertain_nodes)

        rceg_out = self.rceg.graph
        try:
            rceg_out.create_figure("out/prop_two_test_propagation_one.pdf")
        except InvocationException:
            pass
        assert rceg_out.edges["w0", "w1", "a"]["probability"] == (
            pytest.approx(0.46, abs=0.01)
        )
        assert rceg_out.edges["w0", "w2", "b"]["probability"] == (
            pytest.approx(0.54, abs=0.01)
        )
        assert rceg_out.edges["w1", "w4", "d"]["probability"] == (
            pytest.approx(0.35, abs=0.01)
        )
        assert rceg_out.edges["w1", "w4", "e"]["probability"] == (
            pytest.approx(0.65, abs=0.01)
        )
        assert rceg_out.edges["w2", "w4", "f"]["probability"] == (
            pytest.approx(0.29, abs=0.01)
        )
        assert rceg_out.edges["w2", "w5", "g"]["probability"] == (
            pytest.approx(0.71, abs=0.01)
        )
        assert rceg_out.edges["w4", "w_infinity", "j"]["probability"] == (
            pytest.approx(1.00, abs=0.01)
        )
        assert rceg_out.edges["w5", "w_infinity", "l"]["probability"] == (
            pytest.approx(0.70, abs=0.01)
        )
        assert rceg_out.edges["w6", "w_infinity", "m"]["probability"] == (
            pytest.approx(1.00, abs=0.01)
        )
        assert rceg_out.edges["w5", "w6", "o"]["probability"] == (
            pytest.approx(0.30, abs=0.01)
        )

    def test_propagation_two(self):
        uncertain_nodes = [
            {"w1", "w2"},
            {"w5", "w6"},
        ]

        self.rceg.add_uncertain_node_set_list(uncertain_nodes)
        rceg_out = self.rceg.graph
        try:
            rceg_out.create_figure("out/prop_two_test_propagation_two.pdf")
        except InvocationException:
            pass

        assert rceg_out.edges["w0", "w1", "a"]["probability"] == (
            pytest.approx(0.14, abs=0.01)
        )
        assert rceg_out.edges["w0", "w2", "b"]["probability"] == (
            pytest.approx(0.86, abs=0.01)
        )
        assert rceg_out.edges["w1", "w4", "d"]["probability"] == (
            pytest.approx(0.35, abs=0.01)
        )
        assert rceg_out.edges["w1", "w4", "e"]["probability"] == (
            pytest.approx(0.65, abs=0.01)
        )
        assert rceg_out.edges["w2", "w4", "f"]["probability"] == (
            pytest.approx(0.06, abs=0.01)
        )
        assert rceg_out.edges["w2", "w5", "g"]["probability"] == (
            pytest.approx(0.55, abs=0.01)
        )
        assert rceg_out.edges["w2", "w6", "h"]["probability"] == (
            pytest.approx(0.24, abs=0.01)
        )
        assert rceg_out.edges["w3", "w6", "i"]["probability"] == (
            pytest.approx(1.0, abs=0.01)
        )
        assert rceg_out.edges["w4", "w5", "k"]["probability"] == (
            pytest.approx(1.0, abs=0.01)
        )
        assert rceg_out.edges["w5", "w_infinity", "l"]["probability"] == (
            pytest.approx(1.0, abs=0.01)
        )
        assert rceg_out.edges["w6", "w_infinity", "m"]["probability"] == (
            pytest.approx(1.0, abs=0.01)
        )
        assert rceg_out.edges["w2", "w3", "n"]["probability"] == (
            pytest.approx(0.16, abs=0.01)
        )

    def test_propagation_three(self):
        uncertain_edges = {
            ("w4", "w5", "k"),
            ("w5", "w_infinity", "l"),
        }

        self.rceg.add_uncertain_edge_set(uncertain_edges)
        rceg_out = self.rceg.graph
        try:
            rceg_out.create_figure("out/prop_two_test_propagation_three.pdf")
        except InvocationException:
            pass
