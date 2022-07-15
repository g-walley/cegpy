"""Tests ChainEventGraph"""
# pylint: disable=protected-access
import re
from pathlib import Path
from typing import Dict, List, Mapping, Tuple
import unittest
from unittest.mock import Mock, patch
import networkx as nx
import pandas as pd
import pytest
import pytest_mock
from cegpy import StagedTree, ChainEventGraph
from cegpy.graphs._ceg import (
    CegAlreadyGenerated,
    _merge_edge_data,
)


class TestMockedCEGMethods:
    """Tests that Mock functions in ChainEventGraph"""

    node_prefix = "w"
    sink_suffix = "&infin;"
    staged: StagedTree

    def setup(self):
        """Test setup"""
        df_path = (
            Path(__file__)
            .resolve()
            .parent.parent.joinpath("../data/medical_dm_modified.xlsx")
        )

        self.staged = StagedTree(
            dataframe=pd.read_excel(df_path), name="medical_staged"
        )
        self.staged.calculate_AHC_transitions()

    def test_generate_argument(self, mocker: pytest_mock.MockerFixture):
        """When ChainEventGraph called with generate, the .generate()
        method is called."""
        mocker.patch("cegpy.graphs._ceg.ChainEventGraph.generate")
        ceg = ChainEventGraph(self.staged, generate=True)
        ceg.generate.assert_called_once()  # pylint: disable=no-member


class TestUnitCEG(unittest.TestCase):
    """More ChainEventGraph tests"""

    def setUp(self):
        self.node_prefix = "w"
        self.sink_suffix = "&infin;"
        df_path = (
            Path(__file__)
            .resolve()
            .parent.parent.joinpath("../data/medical_dm_modified.xlsx")
        )

        self.staged = StagedTree(
            dataframe=pd.read_excel(df_path), name="medical_staged"
        )
        self.staged.calculate_AHC_transitions()

    def test_stages_property(self):
        """Stages is a mapping of stage names to lists of nodes"""
        ceg = ChainEventGraph(self.staged, generate=False)
        node_stage_mapping: Mapping = dict(ceg.nodes(data="stage", default=None))
        stages = ceg.stages
        self.assertEqual(len(ceg.nodes), sum(len(nodes) for nodes in stages.values()))
        for node, stage in node_stage_mapping.items():
            self.assertIn(node, stages[stage])

    def test_create_figure(self):
        """.create_figure() called with no filename"""
        ceg = ChainEventGraph(self.staged, generate=False)
        with self.assertLogs("cegpy", level="INFO") as log_cm:
            assert ceg.create_figure() is None
        self.assertEqual(
            ["WARNING:cegpy.chain_event_graph:No filename. Figure not saved."],
            log_cm.output,
        )


class TestCEGHelpersTestCases(unittest.TestCase):
    """Tests some of the CEG helper functions."""

    def setUp(self):
        self.graph = nx.MultiDiGraph()
        self.init_nodes = ["w0", "w1", "w2", "w3", "w4", "w_infinity"]
        self.init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", "w3", "c"),
            ("w1", "w4", "d"),
            ("w2", "w3", "c"),
            ("w2", "w4", "d"),
            ("w3", "w_infinity", "e"),
            ("w4", "w_infinity", "f"),
        ]
        self.graph.root = "w0"
        self.graph.add_nodes_from(self.init_nodes)
        self.graph.add_edges_from(self.init_edges)
        self.ceg = ChainEventGraph(self.graph, generate=False)

    def test_merge_edge_data(self):
        """Edges are merged"""
        edge_1 = dict(
            zip(
                ["count", "prior", "posterior", "probability"],
                [250, 0.5, 250, 0.8],
            )
        )
        edge_2 = dict(
            zip(
                ["count", "prior", "posterior", "probability"],
                [550, 25, 0.4, 0.9],
            )
        )

        new_edge_dict = _merge_edge_data(edge_1=edge_1, edge_2=edge_2)
        self.assert_edges_merged(new_edge_dict, edge_1, edge_2)

    def test_merge_edge_data_with_missing(self):
        """Edges are merged even when some data is missing in one edge."""
        edge_1 = dict(
            zip(
                ["count", "prior"],
                [250, 0.5],
            )
        )
        edge_2 = dict(
            zip(
                ["count", "prior", "posterior"],
                [550, 25, 0.4],
            )
        )

        new_edge_dict = _merge_edge_data(edge_1=edge_1, edge_2=edge_2)
        self.assert_edges_merged(new_edge_dict, edge_1, edge_2)

    def assert_edges_merged(self, new_edge: Dict, edge_1: Dict, edge_2: Dict):
        """Edges were merged successfully."""

        self.assertSetEqual(
            set(new_edge.keys()),
            set(edge_1.keys()).union(set(edge_2.keys())),
            msg="Edges do not have the same keys.",
        )

        for key, value in new_edge.items():
            if key == "probability":
                self.assertEqual(
                    value, edge_1.get(key), msg="Probability shouldn't be summed."
                )
            else:
                self.assertEqual(
                    value,
                    (edge_1.get(key, 0) + edge_2.get(key, 0)),
                    msg=f"{key} not merged. Merged value: {value}",
                )

    def test_relabel_nodes(self):
        """Relabel nodes successfully renames all the nodes."""
        self.ceg._relabel_nodes()
        node_pattern = r"^w([0-9]+)|w(_infinity)$"
        prog = re.compile(node_pattern)
        for node in self.ceg.nodes:
            result = prog.match(node)
            assert result is not None, "Node does not match expected format."

    def test_merge_outgoing_edges(self):
        """Outgoing edges are merged."""
        edges_to_add = [
            {
                "src": "s1",
                "dst": "s3",
                "key": "event_1",
                "counts": 5,
                "priors": 0.2,
                "posteriors": 1,
                "probability": 0.8,
            },
            {
                "src": "s1",
                "dst": "s4",
                "key": "event_2",
                "counts": 10,
                "priors": 0.3,
                "posteriors": 5,
                "probability": 0.2,
            },
            {
                "src": "s2",
                "dst": "s3",
                "key": "event_1",
                "counts": 11,
                "priors": 0.5,
                "posteriors": 2,
                "probability": 0.8,
            },
            {
                "src": "s2",
                "dst": "s4",
                "key": "event_2",
                "counts": 6,
                "priors": 0.9,
                "posteriors": 3,
                "probability": 0.2,
            },
        ]
        for edge in edges_to_add:
            self.ceg.add_edge(
                u_for_edge=edge["src"],
                v_for_edge=edge["dst"],
                key=edge["key"],
                counts=edge["counts"],
                priors=edge["priors"],
                posteriors=edge["posteriors"],
            )
        self.ceg.remove_edges_from(
            [
                ("s1", "s3", "c"),
                ("s1", "s4", "d"),
                ("s2", "s3", "c"),
                ("s2", "s4", "d"),
            ]
        )

        expected = [(edge["src"], edge["dst"], edge["key"]) for edge in edges_to_add]
        actual = self.ceg._merge_and_add_edges("s99", "s1", "s2")
        for edge in expected:
            assert (
                edge in actual
            ), f"Expected {edge} in return value, but it was not found."
        assert len(expected) == len(
            actual
        ), "Actual number of edges does not match expected number of edges."


class TestNodesCanBeMerged(unittest.TestCase):
    """Tests nodes_can_be_merged() function."""

    def setUp(self):
        self.graph = nx.MultiDiGraph()
        self.init_nodes = ["w0", "w1", "w2", "w3", "w4", "w_infinity"]
        self.graph.root = "w0"
        self.graph.add_nodes_from(self.init_nodes)

    def test_check_nodes_can_be_merged(self):
        """Nodes can be merged."""
        init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", "w3", "c"),
            ("w1", "w4", "d"),
            ("w2", "w3", "c"),
            ("w2", "w4", "d"),
            ("w3", "w_infinity", "e"),
            ("w4", "w_infinity", "f"),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph, generate=False)
        ceg.nodes["w1"]["stage"] = 2
        ceg.nodes["w2"]["stage"] = 2
        assert ceg._check_nodes_can_be_merged("w1", "w2"), "Nodes should be mergeable."

    def test_nodes_not_in_same_stage_cannot_be_merged(self):
        """Nodes not in same stage cannot be merged"""
        init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", "w3", "c"),
            ("w1", "w4", "d"),
            ("w2", "w3", "c"),
            ("w2", "w4", "d"),
            ("w3", "w_infinity", "e"),
            ("w4", "w_infinity", "f"),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph, generate=False)
        ceg.nodes["w1"]["stage"] = 1
        ceg.nodes["w2"]["stage"] = 2
        assert not ceg._check_nodes_can_be_merged(
            "w1", "w2"
        ), "Nodes should not be mergeable."

    def test_nodes_with_different_successor_nodes(self):
        """Nodes with different successor nodes won't be merged."""
        init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", "w3", "c"),
            ("w1", "w4", "d"),
            ("w2", "w_infinity", "c"),
            ("w2", "w4", "d"),
            ("w3", "w_infinity", "e"),
            ("w4", "w_infinity", "f"),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph, generate=False)
        ceg.nodes["w1"]["stage"] = 2
        ceg.nodes["w2"]["stage"] = 2
        assert not ceg._check_nodes_can_be_merged(
            "w1", "w2"
        ), "Nodes should not be mergeable."

    def test_nodes_with_different_outgoing_edges(self):
        """Nodes with different outgoing edges won't be merged."""
        init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", "w3", "c"),
            ("w1", "w4", "d"),
            ("w2", "w3", "g"),
            ("w2", "w4", "d"),
            ("w3", "w_infinity", "e"),
            ("w4", "w_infinity", "f"),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph, generate=False)
        ceg.nodes["w1"]["stage"] = 2
        ceg.nodes["w2"]["stage"] = 2
        assert not ceg._check_nodes_can_be_merged(
            "w1", "w2"
        ), "Nodes should not be mergeable."

    def test_merging_of_nodes(self):
        """The nodes are merged, and all edges are merged too."""
        init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", "w3", "c"),
            ("w1", "w4", "d"),
            ("w2", "w3", "c"),
            ("w2", "w4", "d"),
            ("w3", "w_infinity", "e"),
            ("w4", "w_infinity", "f"),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph, generate=False)
        ceg.nodes["w1"]["stage"] = 2
        ceg.nodes["w2"]["stage"] = 2
        ceg._merge_nodes({("w1", "w2")})
        expected_edges = [
            ("w0", "w1", "a"),
            ("w0", "w1", "b"),
            ("w1", "w3", "c"),
            ("w1", "w4", "d"),
            ("w3", "w_infinity", "e"),
            ("w4", "w_infinity", "f"),
        ]
        for edge in expected_edges:
            self.assertIn(edge, list(ceg.edges))

    def test_merging_of_three_nodes(self):
        """The three nodes are merged, and all edges are merged too."""
        self.graph.add_node("w5")
        init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w0", "w3", "c"),
            ("w1", "w4", "d"),
            ("w1", "w5", "e"),
            ("w2", "w4", "d"),
            ("w2", "w5", "e"),
            ("w3", "w4", "d"),
            ("w3", "w5", "e"),
            ("w4", "w_infinity", "f"),
            ("w5", "w_infinity", "g"),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph, generate=False)

        nodes_to_merge = {"w1", "w2", "w3"}
        ceg.nodes["w1"]["stage"] = 2
        ceg.nodes["w2"]["stage"] = 2
        ceg.nodes["w3"]["stage"] = 2
        ceg._merge_nodes({("w1", "w2"), ("w2", "w3"), ("w1", "w3")})
        nodes_post_merge = set(ceg.nodes)
        merged_node = nodes_post_merge.intersection(nodes_to_merge).pop()
        expected_edges = [
            ("w0", merged_node, "a"),
            ("w0", merged_node, "b"),
            ("w0", merged_node, "c"),
            (merged_node, "w4", "d"),
            (merged_node, "w5", "e"),
            ("w4", "w_infinity", "f"),
            ("w5", "w_infinity", "g"),
        ]

        for edge in expected_edges:
            self.assertIn(edge, list(ceg.edges))

        self.assertEqual(len(list(ceg.edges)), len(expected_edges))


class TestTrimLeavesFromGraph(unittest.TestCase):
    """Tests trim_leaves_from_graph"""

    def setUp(self):
        self.graph = nx.MultiDiGraph()
        self.init_nodes = ["w0", "w1", "w2", "w3", "w4"]
        self.init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", "w3", "c"),
            ("w1", "w4", "d"),
            ("w2", "w3", "c"),
            ("w2", "w4", "d"),
        ]
        self.graph.root = "w0"
        self.leaves = ["w3", "w4"]
        self.graph.add_nodes_from(self.init_nodes)
        self.graph.add_edges_from(self.init_edges)
        self.ceg = ChainEventGraph(self.graph, generate=False)

    def test_leaves_trimmed_from_graph(self) -> None:
        """Leaves are trimmed from the graph."""
        self.ceg._trim_leaves_from_graph()
        for leaf in self.leaves:
            self.assertIsNone(self.ceg.nodes.get(leaf), "Leaf was not removed.")

            for edge_list_key in self.ceg.edges.keys():
                self.assertNotEqual(
                    edge_list_key[1],
                    leaf,
                    msg=f"Edge still pointing to leaf: {leaf}",
                )

        expected_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", self.ceg.sink, "c"),
            ("w1", self.ceg.sink, "d"),
            ("w2", self.ceg.sink, "c"),
            ("w2", self.ceg.sink, "d"),
        ]
        for edge in expected_edges:
            self.assertIn(
                edge,
                list(self.ceg.edges),
                msg=f"Edge not found: {edge}",
            )

        self.assertEqual(
            len(list(self.ceg.edges)),
            len(expected_edges),
            "Wrong number of edges.",
        )


class TestPathList:
    """Tests path list generation"""

    # pylint: disable=too-few-public-methods
    graph = nx.MultiDiGraph()

    def test_path_list_generation(self):
        """Path list is generated correctly."""

        init_nodes = ["w0", "w1", "w2", "w3", "w4", "w5", "w_infinity"]
        init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", "w3", "e"),
            ("w1", "w4", "f"),
            ("w2", "w_infinity", "c"),
            ("w3", "w_infinity", "d"),
            ("w4", "w5", "c"),
            ("w5", "w_infinity", "d"),
        ]
        self.graph.root = "w0"
        self.graph.add_nodes_from(init_nodes)
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph, generate=False)
        actual_path_list = ceg.path_list
        expected_paths = [
            [("w0", "w1", "a"), ("w1", "w3", "e"), ("w3", "w_infinity", "d")],
            [
                ("w0", "w1", "a"),
                ("w1", "w4", "f"),
                ("w4", "w5", "c"),
                ("w5", "w_infinity", "d"),
            ],
            [("w0", "w2", "b"), ("w2", "w_infinity", "c")],
        ]
        for path in expected_paths:
            assert path in actual_path_list, f"Path not found: {path}"

        assert len(actual_path_list) == len(
            expected_paths
        ), "Incorrect number of paths."


class TestDistanceToSink(unittest.TestCase):
    """Tests distance to sink calculation"""

    def setUp(self):
        self.graph = nx.MultiDiGraph()
        self.init_nodes = ["w0", "w1", "w2", "w3", "w4", "w5", "w_infinity"]
        self.init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", "w3", "e"),
            ("w1", "w4", "f"),
            ("w2", "w_infinity", "c"),
            ("w3", "w_infinity", "d"),
            ("w4", "w5", "c"),
            ("w5", "w_infinity", "d"),
        ]
        self.graph.add_nodes_from(self.init_nodes)
        self.graph.add_edges_from(self.init_edges)
        self.graph.root = "w0"
        self.ceg = ChainEventGraph(self.graph, generate=False)

    def test_update_distances_to_sink(self) -> None:
        """Distance to sink is always max length of paths to sink."""

        def check_distances():
            actual_node_dists = nx.get_node_attributes(self.ceg, "max_dist_to_sink")
            for node, distance in actual_node_dists.items():
                assert distance == expected_node_dists[node]

        expected_node_dists = {
            "w0": 4,
            "w1": 3,
            "w2": 1,
            "w3": 1,
            "w4": 2,
            "w5": 1,
            "w_infinity": 0,
        }
        self.ceg._update_distances_to_sink()
        check_distances()

        # Add another edge to the dictionary, to show that the path is max,
        # and not min distance to sink
        self.ceg.add_edge("w1", self.ceg.sink)
        self.ceg.add_edge("w4", self.ceg.sink)
        self.ceg._update_distances_to_sink()
        check_distances()

    def test_gen_nodes_with_increasing_distance(self) -> None:
        """Tests generate_nodes_with_increasing_distance"""
        expected_nodes = {
            0: [self.ceg.sink],
            1: ["w2", "w3", "w5"],
            2: ["w4"],
            3: ["w1"],
            4: [self.ceg.root],
        }
        for dist, nodes in expected_nodes.items():
            for node in nodes:
                self.ceg.nodes[node]["max_dist_to_sink"] = dist

        nodes_gen = self.ceg._gen_nodes_with_increasing_distance(start=0)

        for dist, nodes in expected_nodes.items():
            actual_node_list = next(nodes_gen)
            self.assertEqual(actual_node_list.sort(), nodes.sort())


class TestEdgeInfoAttributes:
    """Test edge_info argument."""

    med_s_z_paths: List[Tuple]
    med_df: pd.DataFrame
    med_st: StagedTree

    def setup(self):
        """Test Setup"""
        med_df_path = (
            Path(__file__)
            .resolve()
            .parent.parent.joinpath("../data/medical_dm_modified.xlsx")
        )
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_st = StagedTree(
            dataframe=self.med_df, sampling_zero_paths=self.med_s_z_paths
        )

    def test_figure_with_wrong_edge_attribute(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Ensures a warning is raised when a non-existent
        attribute is passed for the edge_info argument"""
        msg = (
            r"edge_info 'prob' does not exist for the "
            r"ChainEventGraph class. Using the default of 'probability' values "
            r"on edges instead. For more information, see the "
            r"documentation."
        )

        # stratified medical dataset
        ceg = ChainEventGraph(self.med_st, generate=False)
        _ = ceg.create_figure(filename=None, edge_info="prob")
        assert msg in caplog.text, "Expected log message not logged."


@patch.object(ChainEventGraph, "_relabel_nodes")
@patch.object(ChainEventGraph, "_gen_nodes_with_increasing_distance")
@patch.object(ChainEventGraph, "_backwards_construction")
@patch.object(ChainEventGraph, "_update_distances_to_sink")
@patch.object(ChainEventGraph, "_trim_leaves_from_graph")
@patch.object(nx, "relabel_nodes")
class TestGenerate(unittest.TestCase):
    """Tests the .generate() method"""

    def setUp(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.init_nodes = [
            "s0",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6",
            "s7",
            "s8",
            "s9",
            "s10",
            "s11",
            "s12",
        ]
        self.init_edges = [
            ("s0", "s1", "a"),
            ("s0", "s2", "b"),
            ("s1", "s3", "c"),
            ("s1", "s4", "d"),
            ("s2", "s5", "e"),
            ("s2", "s6", "f"),
            ("s3", "s7", "g"),
            ("s3", "s8", "h"),
            ("s4", "s9", "i"),
            ("s4", "s10", "j"),
            ("s4", "s11", "k"),
            ("s0", "s12", "l"),
        ]
        self.graph.add_nodes_from(self.init_nodes)
        self.graph.add_edges_from(self.init_edges)
        self.graph.root = "s0"
        self.graph.ahc_output = {
            "Merged Situations": [("s1", "s2")],
            "Loglikelihood": 1234.5678,
        }
        self.ceg = ChainEventGraph(self.graph, generate=False)

    def test_raises_exception_when_called_twice(self, *_):
        """.generate() raises a CegAlreadyGenerated error when called twice"""
        self.ceg.generate()
        with self.assertRaises(CegAlreadyGenerated):
            self.ceg.generate()

    def test_raises_exception_when_there_is_no_ahc_output(self, *_):
        """.generate() raises a ValueError when ahc_output doesn't exist"""
        self.ceg.ahc_output = None
        with self.assertRaises(
            ValueError, msg="There is no AHC output in your StagedTree."
        ):
            self.ceg.generate()

    # pylint: disable=too-many-arguments
    def test_calls_helper_functions_in_the_correct_order(
        self,
        nx_relabel: Mock,
        trim_leaves: Mock,
        update_distances: Mock,
        backwards_construction: Mock,
        gen_nodes: Mock,
        relabel_nodes: Mock,
    ):
        """.generate() calls the helper functions"""
        self.ceg.generate()
        nx_relabel.assert_called_once_with(
            self.ceg, {self.ceg.staged_root: self.ceg.root}, copy=False
        )
        trim_leaves.assert_called_once_with()
        update_distances.assert_called_once_with()
        backwards_construction.assert_called_once_with(gen_nodes.return_value)
        gen_nodes.assert_called_once_with(start=1)
        relabel_nodes.assert_called_once_with()

    def test_generated_flag_set(self, *_):
        """The generated flag is set to True when .generate() is called"""
        self.ceg.generate()
        assert self.ceg.generated is True


class TestBackwardsConstruction(unittest.TestCase):
    """Tests the ._backwards_construction() method"""

    @staticmethod
    def gen_sets_of_nodes():
        """Generates a set of nodes for backwards construction"""
        nodes = [[f"w{i}", f"w{i+1}"] for i in range(5, 1, -1)]
        for node in nodes:
            yield node

    def test_backwards_construction_always_ends(self):
        """
        ._backwards_construction() always ends, even if there are no
        nodes to process
        """
        graph = nx.MultiDiGraph()
        init_nodes = [
            "w0",
            "w1",
            "w2",
            "w3",
            "w4",
            "w5",
            "w6",
            "w7",
            "w8",
            "w9",
            "w10",
            "w11",
            "w12",
        ]
        init_edges = [
            ("w0", "w1", "a"),
            ("w0", "w2", "b"),
            ("w1", "w3", "c"),
            ("w1", "w4", "d"),
            ("w2", "w5", "e"),
            ("w2", "w6", "f"),
            ("w3", "w7", "g"),
            ("w3", "w8", "h"),
            ("w4", "w9", "i"),
            ("w4", "w10", "j"),
            ("w4", "w11", "k"),
            ("w0", "w12", "l"),
        ]
        graph.add_nodes_from(init_nodes)
        graph.add_edges_from(init_edges)
        graph.root = "w0"
        graph.ahc_output = {
            "Merged Situations": [("s1", "s2")],
            "Loglikelihood": 1234.5678,
        }
        ceg = ChainEventGraph(graph, generate=False)

        ceg._backwards_construction(self.gen_sets_of_nodes())

    def test_backwards_construction_produces_ceg(self):
        """The backwards construction algorithm takes the staged tree and
        makes the ceg."""
        nodes = ["w0", "s1", "s2", "s3", "s4", "s5", "s11", "s12", "s13", "w_infinity"]
        edges = [
            ("w0", "s1", "hospital"),
            ("w0", "s2", "community"),
            ("s1", "s3", "test"),
            ("s1", "s4", "no test"),
            ("s2", "s11", "test"),
            ("s2", "s12", "no test"),
            ("s3", "s5", "positive"),
            ("s3", "w_infinity", "negative"),
            ("s4", "w_infinity", "death"),
            ("s4", "w_infinity", "recovery"),
            ("s5", "w_infinity", "death"),
            ("s5", "w_infinity", "recovery"),
            ("s11", "s13", "positive"),
            ("s11", "w_infinity", "negative"),
            ("s12", "w_infinity", "death"),
            ("s12", "w_infinity", "recovery"),
            ("s13", "w_infinity", "death"),
            ("s13", "w_infinity", "recovery"),
        ]
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        graph.root = "w0"
        graph.nodes["w0"]["stage"] = 0
        graph.nodes["s1"]["stage"] = 1
        graph.nodes["s2"]["stage"] = 2
        graph.nodes["s3"]["stage"] = 3
        graph.nodes["s4"]["stage"] = 4
        graph.nodes["s5"]["stage"] = 4
        graph.nodes["s11"]["stage"] = 3
        graph.nodes["s12"]["stage"] = 5
        graph.nodes["s13"]["stage"] = 4

        def gen_sets_of_nodes():
            yield ["s4", "s5", "s12", "s13"]
            yield ["s3", "s11"]
            yield ["s1", "s2"]
            yield ["w0"]

        ceg = ChainEventGraph(graph, generate=False)
        ceg._backwards_construction(gen_sets_of_nodes())
        all_nodes = set(ceg.nodes)
        self.assertEqual(len(all_nodes.intersection({"s5", "s4", "s13"})), 1)
        self.assertEqual(len(all_nodes.intersection({"s11", "s3"})), 1)
