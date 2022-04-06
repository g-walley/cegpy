from copy import deepcopy
import networkx as nx
from typing import List, Set, Tuple
from . import ChainEventGraph


class TransporterChainEventGraph:
    def __init__(self, ceg: ChainEventGraph):
        self._ceg = ceg

        self._certain_edges = []
        self._uncertain_edges = []
        self._certain_nodes = set()
        self._uncertain_nodes = []

        self._path_list = []

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
                base_str += '   Edges = []\n'
            else:
                base_str += '   Edges = [\n'
                for edge in edges:
                    base_str += '     %s,\n' % (str(edge))
                base_str += '   ]\n'

            if nodes == set():
                base_str += '   Nodes = {}\n'
            else:
                base_str += '   Nodes = {\n'
                for node in nodes:
                    base_str += "     '%s',\n" % (str(node))
                base_str += '   }\n\n'
            return base_str

        base_str = 'The evidence you have given is as follows:\n'
        base_str += ' Evidence you are certain of:\n'
        base_str = evidence_str(
            base_str,
            self.certain_edges,
            self.certain_nodes
        )
        base_str += ' Evidence you are uncertain of:\n'
        base_str = evidence_str(
            base_str,
            self.uncertain_edges,
            self.uncertain_nodes
        )
        return base_str

    @property
    def certain_edges(self) -> List[Tuple[str]]:
        return self._certain_edges

    @property
    def uncertain_edges(self) -> List[Tuple[str]]:
        return self._uncertain_edges

    @property
    def certain_nodes(self) -> Set[str]:
        return self._certain_nodes

    @property
    def uncertain_nodes(self) -> List[Set[str]]:
        return self._uncertain_nodes

    @property
    def paths(self):
        return self._path_list

    @property
    def reduced(self):
        return self._create_reduced_graph()

    def clear_all_evidence(self):
        """Clears all evidence provided."""
        self._certain_edges = []
        self._uncertain_edges = []
        self._certain_nodes = set()
        self._uncertain_nodes = []

    def add_certain_edge(self, u: str, v: str, label: str):
        """Specify an edge that has been observed."""
        edge = (u, v, label)
        if edge not in self._ceg.edges:
            raise ValueError(
                f"This edge {edge}, does not exist"
                f" in the Chain Event Graph."
            )
        self._certain_edges.append(edge)

    def remove_certain_edge(self, u: str, v: str, label: str):
        """Specify an edge to remove from the certain edges."""
        try:
            edge = (u, v, label)
            self._certain_edges.remove(edge)
        except ValueError:
            raise ValueError(
                f"Edge {(u, v, label)} not found in the certain "
                f"edge list."
            )

    def add_certain_edge_list(self, edges: List[Tuple[str]]):
        """Specify a list of edges that have all been observed.
        E.g. edges = [
            ("s0","s1", "foo"),
            ("s1","s5", "bar"), ...]"""
        for edge in edges:
            self.add_certain_edge(*edge)

    def remove_certain_edge_list(self, edges: List[Tuple[str]]):
        """Specify a list of edges that in the certain edge list
        to remove.
        E.g. edges = [
        ("s0","s1", "foo"),
        ("s1","s5", "bar"), ...]"""
        for edge in edges:
            self.remove_certain_edge(*edge)

    def add_uncertain_edge_set(self, edge_set: Set[Tuple[str]]):
        """Specify a set of edges where one of the edges has
        occured, but you are uncertain of which one it is."""
        for edge in edge_set:
            if edge not in self._ceg.edges:
                raise ValueError(
                    f"The edge {edge}, does not exist"
                    f" in the Chain Event Graph."
                )

        self._uncertain_edges.append(edge_set)

    def remove_uncertain_edge_set(self, edge_set: Set[Tuple[str]]):
        """Specify a set of edges to remove from the uncertain edges."""
        try:
            self._uncertain_edges.remove(edge_set)
        except ValueError:
            raise ValueError(
                f"{edge_set} not found in the uncertain edge list."
            )

    def add_uncertain_edge_set_list(self, edge_sets: List[Set[Tuple[str]]]):
        """Specify a list of sets of edges where one of the edges has
        occured, but you are uncertain of which one it is."""
        for edge_set in edge_sets:
            self.add_uncertain_edge_set(edge_set)

    def remove_uncertain_edge_set_list(self, edge_sets: List[Set[Tuple[str]]]):
        """Specify a list of sets of edges to remove from the evidence list."""
        for edge_set in edge_sets:
            self.remove_uncertain_edge_set(edge_set)

    def add_certain_node(self, node: str):
        """Specify a node that has been observed."""
        if node not in self._ceg.nodes:
            raise ValueError(
                f"The node {node}, does not exist in the Chain Event Graph."
            )
        self._certain_nodes.add(node)

    def remove_certain_node(self, node: str):
        """Specify a node to be removed from the certain nodes list."""
        try:
            self._certain_nodes.remove(node)
        except KeyError:
            raise ValueError(
                f"Node {node} not found in the set of certain nodes."
            )

    def add_certain_node_set(self, nodes: Set[str]):
        """Specify a set of nodes that have been observed."""
        for node in nodes:
            self.add_certain_node(node)

    def remove_certain_node_set(self, nodes: Set[str]):
        """Specify a list of nodes to remove from the list of nodes that have
        been observed."""
        for node in nodes:
            self.remove_certain_node(node)

    def add_uncertain_node_set(self, node_set: Set[str]):
        """Specify a set of nodes where one of the nodes has
        occured, but you are uncertain of which one it is."""
        for node in node_set:
            if node not in self._ceg.nodes:
                raise ValueError(
                    f"The node {node}, does not exist"
                    f" in the Chain Event Graph."
                )

        self._uncertain_nodes.append(node_set)

    def remove_uncertain_node_set(self, node_set: Set[str]):
        """Specify a set of nodes to be removed from the uncertain
        nodes set list."""
        try:
            self._uncertain_nodes.remove(node_set)
        except ValueError:
            raise ValueError(
                f"{node_set} not found in the uncertain node list."
            )

    def add_uncertain_node_set_list(self, node_sets: List[Set[str]]):
        """Specify a list of sets of nodes where in each set, one of
        the nodes has occured, but you are uncertain of which one it is."""
        for node_set in node_sets:
            self.add_uncertain_node_set(node_set)

    def remove_uncertain_node_set_list(self, node_sets: List[Set[str]]):
        """Specify a list of sets nodes to remove from the list of uncertain
        sets of nodes."""
        for node_set in node_sets:
            self.remove_uncertain_node_set(node_set)

    @staticmethod
    def _remove_paths(paths: List, to_remove: List):
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
                for (u, v, _) in path:
                    if node == u or node == v:
                        break
                else:
                    if path not in to_remove:
                        to_remove.append(path)

        self._remove_paths(paths, to_remove)

    def _apply_uncertain_edges(self, paths):
        to_remove = []
        for edge_set in self.uncertain_edges:
            for path in paths:
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

        self._remove_paths(paths, to_remove)

    def _apply_uncertain_nodes(self, paths):
        to_remove = []
        for node_set in self.uncertain_nodes:
            for path in paths:
                node_found = False
                for (u, v, _) in path:
                    if (u in node_set):
                        if node_found:
                            if path not in to_remove:
                                to_remove.append(path)
                            break
                        node_found = True
                else:
                    if not node_found and path not in to_remove:
                        to_remove.append(path)

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
            for (u, v, k) in path:
                edges.add((u, v, k))
                vertices.add(u)
                vertices.add(v)

        self.edges = edges
        self.vertices = vertices

    def _filter_edge(self, u, v, k) -> bool:
        if (u, v, k) in self.edges:
            return True
        else:
            return False

    def _filter_node(self, n) -> bool:
        if n in self.vertices:
            return True
        else:
            return False

    def _propagate_reduced_graph_probabilities(
        self,
        graph: ChainEventGraph
    ) -> None:
        sink = graph.sink_node
        root = graph.root_node
        graph.nodes[sink]['emphasis'] = 1
        node_set = set([sink])

        while node_set != {root}:
            try:
                v_node = node_set.pop()
            except KeyError:
                raise KeyError(
                    "Graph has more than one root..."
                    "Propagation cannot continue.")

            for u_node, edges in graph.pred[v_node].items():
                for edge_label, data in edges.items():
                    edge = (u_node, v_node, edge_label)
                    emph = graph.nodes[v_node]['emphasis']
                    prob = data['probability']
                    graph.edges[edge]['potential'] = \
                        prob * emph
                successors = graph.succ[u_node]

                try:
                    emphasis = 0
                    for _, edges in successors.items():
                        for edge_label, data in edges.items():
                            emphasis += data['potential']

                    graph.nodes[u_node]['emphasis'] = emphasis
                    node_set.add(u_node)
                except KeyError:
                    pass

        for (u, v, k) in list(graph.edges):
            edge_potential = graph.edges[(u, v, k)]['potential']
            pred_node_emphasis = graph.nodes[u]['emphasis']
            probability = edge_potential / pred_node_emphasis
            graph.edges[(u, v, k)]['probability'] = probability

    def _create_reduced_graph(self) -> ChainEventGraph:
        self._update_path_list()
        self._update_edges_and_vertices()
        subgraph = ChainEventGraph(
            nx.subgraph_view(
                G=self._ceg,
                filter_node=self._filter_node,
                filter_edge=self._filter_edge
            )
        ).copy()

        self._propagate_reduced_graph_probabilities(subgraph)

        return subgraph
    # Deal with uncertain edges that are in the same path
