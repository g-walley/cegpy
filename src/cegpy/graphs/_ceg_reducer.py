"""Reduced Chain Event Graph"""

from copy import deepcopy
from typing import Iterable, List, Set, Tuple
import networkx as nx

from cegpy.graphs._ceg import ChainEventGraph


class ChainEventGraphReducer:
    """
    Reduces Chain Event Graphs given certain and/or uncertain evidence.

    :param ceg: Chain event graph object to reduce.
    :type ceg: ChainEventGraph
    """

    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    _path_list: List[List[Tuple[str]]]

    def __init__(self, ceg: ChainEventGraph):
        self._ceg = ceg
        self._path_list = ceg.path_list
        self._certain_edges = []
        self._uncertain_edges = []
        self._certain_nodes = set()
        self._uncertain_nodes = []
        self._edges = set()
        self._vertices = set()

    def __repr__(self) -> str:
        return (
            f"Evidence(certain_edges={str(self.certain_edges)}, "
            f"certain_nodes={str(self.certain_nodes)}, "
            f"uncertain_edges={str(self.uncertain_edges)}, "
            f"uncertain_nodes={str(self.uncertain_nodes)})"
        )

    def __str__(self) -> str:
        """Returns human readable version of the evidence you've provided."""

        def evidence_str(base_str, edges, nodes):
            if edges == []:
                base_str += "   Edges = []\n"
            else:
                base_str += "   Edges = [\n"
                for edge in edges:
                    base_str += f"     {str(edge)},\n"
                base_str += "   ]\n"

            if nodes == set():
                base_str += "   Nodes = {}\n"
            else:
                base_str += "   Nodes = {\n"
                for node in nodes:
                    base_str += f"     {str(node)},\n"
                base_str += "   }\n\n"
            return base_str

        base_str = "The evidence you have given is as follows:\n"
        base_str += " Evidence you are certain of:\n"
        base_str = evidence_str(base_str, self.certain_edges, self.certain_nodes)
        base_str += " Evidence you are uncertain of:\n"
        base_str = evidence_str(base_str, self.uncertain_edges, self.uncertain_nodes)
        return base_str

    @property
    def certain_edges(self) -> List[Tuple[str]]:
        """A list of all edges of the ChainEventGraph that have been observed.

        `certain_edges` is a list of edge tuples of the form:
        `[edge_1, edge_2, ... edge_n]`

        Each edge tuple takes the form:
        `("source_node_name", "destination_node_name", "edge_label")`
        """
        return self._certain_edges

    @property
    def uncertain_edges(self) -> List[Tuple[str]]:
        """A list of sets of edges of the ChainEventGraph which might have occured.
        `uncertain_edges` is a list of sets of edge tuples of the form:

        `[{(a, b, label), (a, c, label)}, {(x, y, label), (x, z, label)}]`

        Each edge tuple takes the form:

        `("source_node_name", "destination_node_name", "edge_label")`
        """
        return self._uncertain_edges

    @property
    def certain_nodes(self) -> Set[str]:
        """A set of all nodes of the ChainEventGraph that have been observed.

        `certain_nodes` is a set of nodes of the form:
        `{"node_1", "node_2", "node_3", ... "node_n"}`"""
        return self._certain_nodes

    @property
    def uncertain_nodes(self) -> List[Set[str]]:
        """A list of sets of nodes of the ChainEventGraph where there is uncertainty
        which of the nodes in each set happened.
        `uncertain_nodes` is a list of sets of nodes of the form:
        `[{"node_1", "node_2"}, {"node_3", "node_4"}, ...]`"""
        return self._uncertain_nodes

    @property
    def paths(self) -> List[List[Tuple[str]]]:
        """
        :return: A list of all paths through the reduced ChainEventGraph."""
        return self._path_list

    @property
    def graph(self) -> ChainEventGraph:
        """
        :return: The reduced graph once all evidence has been taken into account.
        :rtype: ChainEventGraph
        """
        self._update_path_list()
        self._update_edges_and_vertices()
        subgraph_view = nx.subgraph_view(
            G=self._ceg, filter_node=self._filter_node, filter_edge=self._filter_edge
        )
        reduced_graph = ChainEventGraph(subgraph_view, generate=False).copy()

        self._propagate_reduced_graph_probabilities(reduced_graph)

        return reduced_graph

    def clear_all_evidence(self) -> None:
        """Resets the evidence provided."""
        self._certain_edges = []
        self._uncertain_edges = []
        self._certain_nodes = set()
        self._uncertain_nodes = []
        self._edges = set()
        self._vertices = set()

    def add_certain_edge(self, src: str, dst: str, label: str) -> None:
        """
        Specify an edge that has been observed.
        :param src: Edge source node label
        :type src: str

        :param dst: Edge destination node label
        :type dst: str

        :param label: Label of certain edge
        :type label: str
        """

        edge = (src, dst, label)
        if edge not in self._ceg.edges:
            raise ValueError(
                f"This edge {edge}, does not exist" f" in the Chain Event Graph."
            )
        self._certain_edges.append(edge)

    def remove_certain_edge(self, src: str, dst: str, label: str) -> None:
        """
        Specify an edge to remove from the certain edges.
        :param src: Edge source node label
        :type src: str

        :param dst: Edge destination node label
        :type dst: str

        :param label: Label of certain edge
        :type label: str
        """
        try:
            edge = (src, dst, label)
            self._certain_edges.remove(edge)
        except ValueError as err:
            raise ValueError(
                f"Edge {(src, dst, label)} not found in the certain " f"edge list."
            ) from err

    def add_certain_edge_list(self, edges: List[Tuple[str]]) -> None:
        """
        Specify a list of edges that have all been observed.

        :param edges: List of edge tuples of the form ("src", "dst", "label")
        :type edges: List[Tuple[str]]
        """
        for edge in edges:
            self.add_certain_edge(*edge)

    def remove_certain_edge_list(self, edges: List[Tuple[str]]) -> None:
        """
        Specify a list of edges that in the certain edge list
        to remove.

        :param edges: List of edge tuples of the form ("src", "dst", "label")
        :type edges: List[Tuple[str]]
        """
        for edge in edges:
            self.remove_certain_edge(*edge)

    def add_uncertain_edge_set(self, edge_set: Set[Tuple[str]]) -> None:
        """
        Specify a set of edges where one of the edges has
        occured, but you are uncertain of which one it is.

        :param edge_set: Set of edge tuples of the form ("src", "dst", "label")
        :type edge_set: Set[Tuple[str]]
        """
        for edge in edge_set:
            if edge not in self._ceg.edges:
                raise ValueError(
                    f"The edge {edge}, does not exist" f" in the Chain Event Graph."
                )

        self._uncertain_edges.append(edge_set)

    def remove_uncertain_edge_set(self, edge_set: Set[Tuple[str]]) -> None:
        """
        Specify a set of edges to remove from the uncertain edges.

        :param edge_set: Set of edge tuples of the form ("src", "dst", "label")
        :type edge_set: Set[Tuple[str]]
        """
        try:
            self._uncertain_edges.remove(edge_set)
        except ValueError as err:
            raise ValueError(
                f"{edge_set} not found in the uncertain edge list."
            ) from err

    def add_uncertain_edge_set_list(self, edge_sets: List[Set[Tuple[str]]]) -> None:
        """
        Specify a list of sets of edges where one of the edges has
        occured, but you are uncertain of which one it is.

        :param edge_set: List of sets of edge tuples of the form ("src", "dst", "label")
        :type edge_set: List[Set[Tuple[str]]]
        """
        for edge_set in edge_sets:
            self.add_uncertain_edge_set(edge_set)

    def remove_uncertain_edge_set_list(self, edge_sets: List[Set[Tuple[str]]]) -> None:
        """
        Specify a list of sets of edges to remove from the evidence list.

        :param edge_set: List of sets of edge tuples of the form ("src", "dst", "label")
        :type edge_set: List[Set[Tuple[str]]]
        """
        for edge_set in edge_sets:
            self.remove_uncertain_edge_set(edge_set)

    def add_certain_node(self, node: str) -> None:
        """Specify a node in the graph that has been observed.

        :param node: A node label e.g. "w4"
        :type node: str
        """
        if node not in self._ceg.nodes:
            raise ValueError(
                f"The node {node}, does not exist in the Chain Event Graph."
            )
        self._certain_nodes.add(node)

    def remove_certain_node(self, node: str) -> None:
        """
        Specify a node to be removed from the certain nodes list.

        :param node: A node label e.g. "w4"
        :type node: str
        """
        try:
            self._certain_nodes.remove(node)
        except KeyError as err:
            raise ValueError(
                f"Node {node} not found in the set of certain nodes."
            ) from err

    def add_certain_node_set(self, nodes: Set[str]) -> None:
        """
        Specify a set of nodes that have been observed.

        :param nodes: A set of node labels e.g. {"w4", "w8"}
        :type nodes: Set[str]
        """
        for node in nodes:
            self.add_certain_node(node)

    def remove_certain_node_set(self, nodes: Set[str]) -> None:
        """
        Specify a list of nodes to remove from the list of nodes that have
        been observed.

        :param nodes: A set of node labels e.g. {"w4", "w8"}
        :type nodes: Set[str]
        """
        for node in nodes:
            self.remove_certain_node(node)

    def add_uncertain_node_set(self, node_set: Set[str]) -> None:
        """
        Specify a set of nodes where one of the nodes has
        occured, but you are uncertain of which one it is.

        :param nodes: A set of node labels e.g. {"w4", "w8"}
        :type nodes: Set[str]
        """
        for node in node_set:
            if node not in self._ceg.nodes:
                raise ValueError(
                    f"The node {node}, does not exist" f" in the Chain Event Graph."
                )

        self._uncertain_nodes.append(node_set)

    def remove_uncertain_node_set(self, node_set: Set[str]) -> None:
        """
        Specify a set of nodes to be removed from the uncertain
        nodes set list.

        :param nodes: A set of node labels e.g. {"w4", "w8"}
        :type nodes: Set[str]
        """
        try:
            self._uncertain_nodes.remove(node_set)
        except ValueError as err:
            raise ValueError(
                f"{node_set} not found in the uncertain node list."
            ) from err

    def add_uncertain_node_set_list(self, node_sets: List[Set[str]]) -> None:
        """
        Specify a list of sets of nodes where in each set, one of
        the nodes has occured, but you are uncertain of which one it is.

        :param nodes: A collection of sets of uncertain nodes.
        :type nodes: List[Set[str]]"""
        for node_set in node_sets:
            self.add_uncertain_node_set(node_set)

    def remove_uncertain_node_set_list(self, node_sets: List[Set[str]]) -> None:
        """
        Specify a list of sets nodes to remove from the list of uncertain
        sets of nodes.

        :param nodes: A collection of sets of uncertain nodes.
        :type nodes: List[Set[str]]
        """
        for node_set in node_sets:
            self.remove_uncertain_node_set(node_set)

    @staticmethod
    def _remove_paths(paths: List, to_remove: Iterable):
        for path in to_remove:
            paths.remove(path)

    def _apply_certain_edges(self, paths):
        to_remove = []
        for edge in self.certain_edges:
            for path in paths:
                if (edge not in path) and (path not in to_remove):
                    to_remove.append(path)

        self._remove_paths(paths, to_remove)

    def _apply_certain_nodes(self, paths):
        to_remove = []
        for node in self.certain_nodes:
            for path in paths:
                for (src, dst, _) in path:
                    if node in (src, dst):
                        break
                else:
                    if path not in to_remove:
                        to_remove.append(path)

        self._remove_paths(paths, to_remove)

    def _apply_uncertain_edges(self, paths):
        to_remove = []

        def find_edge_and_add_to_remove():
            edge_found = False
            for edge in path:
                if edge in edge_set:
                    if edge_found:
                        if path not in to_remove:
                            to_remove.append(path)
                        break
                    edge_found = True
            else:
                if not edge_found and path not in to_remove:
                    to_remove.append(path)

        for edge_set in self.uncertain_edges:
            for path in paths:
                find_edge_and_add_to_remove()

        self._remove_paths(paths, to_remove)

    def _apply_uncertain_nodes(self, paths):
        to_remove = []

        def find_node_and_add_to_remove():
            node_found = False
            for (src, _, _) in path:
                if src in node_set:
                    if node_found:
                        if path not in to_remove:
                            to_remove.append(path)
                        break
                    node_found = True
            else:
                if not node_found and path not in to_remove:
                    to_remove.append(path)

        for node_set in self.uncertain_nodes:
            for path in paths:
                find_node_and_add_to_remove()

        self._remove_paths(paths, to_remove)

    def _update_path_list(self):
        path_list = deepcopy(self._ceg.path_list)

        if self.certain_edges:
            self._apply_certain_edges(path_list)

        if self.certain_nodes:
            self._apply_certain_nodes(path_list)

        if self.uncertain_edges:
            self._apply_uncertain_edges(path_list)

        if self.uncertain_nodes:
            self._apply_uncertain_nodes(path_list)

        self._path_list = path_list

    def _update_edges_and_vertices(self):
        edges = set()
        vertices = set()

        for path in self.paths:
            for (src, dst, label) in path:
                edges.add((src, dst, label))
                vertices.add(src)
                vertices.add(dst)

        self._edges = edges
        self._vertices = vertices

    def _filter_edge(self, src, dst, label) -> bool:
        return bool((src, dst, label) in self._edges)

    def _filter_node(self, node) -> bool:
        return bool(node in self._vertices)

    @staticmethod
    def _propagate_reduced_graph_probabilities(graph: ChainEventGraph) -> None:
        # pylint: disable=too-many-locals
        sink = graph.sink
        root = graph.root
        graph.nodes[sink]["emphasis"] = 1
        node_set = set([sink])

        while node_set != {root}:
            try:
                dst = node_set.pop()
            except KeyError as err:
                raise KeyError(
                    "Graph has more than one root... Propagation cannot continue."
                ) from err

            for src, edges in graph.pred[dst].items():
                for edge_label, data in edges.items():
                    edge = (src, dst, edge_label)
                    emph = graph.nodes[dst]["emphasis"]
                    prob = data["probability"]
                    graph.edges[edge]["potential"] = prob * emph
                successors = graph.succ[src]

                try:
                    emphasis = 0
                    for _, succ_edges in successors.items():
                        for edge_label, data in succ_edges.items():
                            emphasis += data["potential"]

                    graph.nodes[src]["emphasis"] = emphasis
                    node_set.add(src)
                except KeyError:
                    pass

        for (src, dst, label) in list(graph.edges):
            edge_potential = graph.edges[(src, dst, label)]["potential"]
            pred_node_emphasis = graph.nodes[src]["emphasis"]
            probability = edge_potential / pred_node_emphasis
            graph.edges[(src, dst, label)]["probability"] = probability
