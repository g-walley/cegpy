"""Chain Event Graph"""

from collections import defaultdict
from copy import deepcopy
import itertools as it
import logging
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)
import pydotplus as pdp
import networkx as nx
from IPython.display import Image
from IPython import get_ipython

from ..utilities.util import Util
from ..trees.staged import StagedTree

logger = logging.getLogger('cegpy.chain_event_graph')


class ChainEventGraph(nx.MultiDiGraph):
    """
    Class: Chain Event Graph

    Input: Staged tree object (StagedTree)
    Output: Chain event graphs
    """

    sink_suffix: str
    node_prefix: str
    path_list: List[str]

    def __init__(
        self,
        staged_tree: Optional[StagedTree] = None,
        generate: bool = False,
        **attr
    ):
        self.ahc_output = deepcopy(getattr(staged_tree, "ahc_output", None))
        super().__init__(staged_tree, **attr)
        self.sink_suffix = "&infin;"
        self.node_prefix = "w"
        self._stages = {}
        self.path_list = []

        if generate:
            self.generate()

    @property
    def sink_node(self) -> str:
        """Sink node name as a string."""
        return f"{self.node_prefix}{self.sink_suffix}"

    @property
    def root_node(self) -> str:
        """Root node name as a string."""
        return f"{self.node_prefix}0"

    @property
    def stages(self) -> Mapping[str, Set[str]]:
        """Mapping of stages to constituent nodes."""
        node_stages = dict(self.nodes(data='stage', default=None))
        stages = defaultdict(set)
        for node, stage in node_stages.items():
            stages[stage].add(node)

        return stages

    def generate(self):
        """
        This function takes the output of the AHC algorithm and identifies
        the positions i.e. the vertices of the CEG and the edges of the CEG
        along with their edge labels and edge counts. Here we use the
        algorithm in our paper with the optimal stopping time.
        """

        if self.ahc_output == {}:
            raise ValueError("Run staged tree AHC transitions first.")
        # rename root node:
        nx.relabel_nodes(self, {'s0': self.root_node}, copy=False)
        _trim_leaves_from_graph(self)
        _update_distances_to_sink(self)
        src_node_gen = _gen_nodes_with_increasing_distance(self, start=1)
        next_set_of_nodes = next(src_node_gen)

        while next_set_of_nodes != [self.root_node]:
            nodes_to_merge = set()
            while len(next_set_of_nodes) > 1:
                node_1 = next_set_of_nodes.pop(0)
                for node_2 in next_set_of_nodes:
                    mergeable = _check_nodes_can_be_merged(
                        self, node_1, node_2
                    )
                    if mergeable:
                        nodes_to_merge.add((node_1, node_2))

            if nodes_to_merge:
                self._merge_nodes(nodes_to_merge)

            try:
                next_set_of_nodes = next(src_node_gen)
            except StopIteration:
                break

        _relabel_nodes(self)
        self._update_path_list()

    def _merge_nodes(self, nodes_to_merge: Set):
        """nodes to merge should be a set of 2 element tuples"""
        temp_1 = 'temp_1'
        temp_2 = 'temp_2'
        while nodes_to_merge:
            nodes = nodes_to_merge.pop()
            new_node = nodes[0]
            # Copy nodes to temp nodes
            node_map = {
                nodes[0]: temp_1,
                nodes[1]: temp_2
            }
            nx.relabel_nodes(self, node_map, copy=False)
            self.add_node(new_node)

            edges_to_remove = _merge_and_add_edges(
                self,
                new_node,
                temp_1,
                temp_2,
            )
            self.remove_edges_from(edges_to_remove)
            nx.relabel_nodes(
                G=self,
                mapping={temp_1: new_node, temp_2: new_node},
                copy=False
            )

            # Some nodes have been removed, we need to update the
            # mergeable list to point to new nodes if required
            temp_list = list(nodes_to_merge)
            for pair in temp_list:
                if nodes[1] in pair:
                    new_pair = (
                        # the other node of the pair
                        pair[pair.index(nodes[1]) - 1],
                        # the new node it will be merged to
                        new_node
                    )
                    nodes_to_merge.remove(pair)
                    if new_pair[0] != new_pair[1]:
                        nodes_to_merge.add(new_pair)

    @property
    def dot_graph(self) -> pdp.Dot:
        """Dot representation of the CEG."""
        return self._generate_dot_graph()

    def _generate_dot_graph(self):
        graph = pdp.Dot(graph_type='digraph', rankdir='LR')
        edge_probabilities = list(
            self.edges(data='probability', default=1, keys=True)
        )

        for (src, dst, label, probability) in edge_probabilities:
            full_label = f"{label}\n{float(probability):.2f}"
            graph.add_edge(
                pdp.Edge(
                    src=src,
                    dst=dst,
                    label=full_label,
                    labelfontcolor='#009933',
                    fontsize='10.0',
                    color='black'
                )
            )
        nodes = list(nx.topological_sort(self))
        for node in nodes:
            try:
                fill_colour = self.nodes[node]['colour']
            except KeyError:
                fill_colour = 'white'
            label = "<" + node[0] + "<SUB>" + node[1:] + "</SUB>" + ">"
            graph.add_node(
                pdp.Node(
                    name=node,
                    label=label,
                    style='filled',
                    fillcolor=fill_colour
                )
            )
        return graph

    def create_figure(self, filename) -> Union[Image, None]:
        """
        Draws the chain event graph representation of the stage tree,
        and saves it to "<filename>.filetype". Supports any filetype that
        graphviz supports. e.g: "event_tree.png" or "event_tree.svg" etc.
        """
        filename, filetype = Util.generate_filename_and_mkdir(filename)
        graph = self.dot_graph
        graph.write(str(filename), format=filetype)

        if get_ipython() is not None:
            # pylint: disable=no-member
            return Image(graph.create_png())

        return None

    def _update_path_list(self) -> None:
        """Updates the path list, should be called after graph is modified."""
        path_generator = nx.all_simple_edge_paths(
            self,
            self.root_node,
            self.sink_node
        )
        path_list = []
        while True:
            try:
                path_list.append(next(path_generator))
            except StopIteration:
                self.path_list = path_list
                break


def _merge_edge_data(
    edge_1: Dict[str, Any],
    edge_2: Dict[str, Any],
) -> Dict[str, Any]:
    """Merges the counts, priors, and posteriors of two edges."""
    new_edge_data = {}
    edge = edge_1 if len(edge_1) > len(edge_2) else edge_2
    for key in edge:
        if key == "probability":
            new_edge_data[key] = edge_1.get(key, 1)
        else:
            new_edge_data[key] = (
                edge_1.get(key, 0) + edge_2.get(key, 0)
            )
    return new_edge_data


def _relabel_nodes(ceg: ChainEventGraph):
    """Relabels nodes whilst maintaining ordering."""
    num_iterator = it.count(1, 1)
    nodes_to_rename = list(ceg.succ[ceg.root_node].keys())
    # first, relabel the successors of this node
    node_mapping = {}
    while nodes_to_rename:
        for node in nodes_to_rename.copy():
            node_mapping[node] = f"{ceg.node_prefix}{next(num_iterator)}"
            for succ in ceg.succ[node].keys():
                if (succ != ceg.sink_node and succ not in nodes_to_rename):
                    nodes_to_rename.append(succ)
            nodes_to_rename.remove(node)

    nx.relabel_nodes(
        ceg,
        node_mapping,
        copy=False
    )


def _merge_and_add_edges(
    ceg: ChainEventGraph,
    new_node: str,
    node_1: str,
    node_2: str,
) -> List[Tuple]:
    """Merges outgoing edges of two nodes so that the two nodes can be
    merged."""
    old_edges_to_remove = []
    for succ, t1_edge_dict in ceg.succ[node_1].items():
        edge_labels = list(t1_edge_dict.keys())
        while edge_labels:
            label = edge_labels.pop(0)
            n1_edge_data = t1_edge_dict[label]
            n2_edge_data = ceg.succ[node_2][succ][label]

            new_edge_data = _merge_edge_data(
                edge_1=n1_edge_data,
                edge_2=n2_edge_data,
            )
            ceg.add_edge(
                u_for_edge=new_node,
                v_for_edge=succ,
                key=label,
                **new_edge_data,
            )
            old_edges_to_remove.extend(
                [(node_1, succ, label), (node_2, succ, label)]
            )

    return old_edges_to_remove


def _trim_leaves_from_graph(ceg: ChainEventGraph):
    """Trims all the leaves from the graph, and points each incoming
    edge to the sink node."""
    # Create new CEG sink node
    ceg.add_node(ceg.sink_node, colour='lightgrey')
    outgoing_edges = deepcopy(ceg.succ).items()
    # Check to see if any nodes have no outgoing edges.
    for node, out_edges in outgoing_edges:
        if not out_edges and node != ceg.sink_node:
            mapping = {node: ceg.sink_node}
            nx.relabel_nodes(ceg, mapping, copy=False)


def _update_distances_to_sink(ceg: ChainEventGraph) -> None:
    """
    Iterates through the graph until it finds the root node.
    For each node, it determines the maximum number of edges
    from that node to the sink node.
    """
    max_dist = "max_dist_to_sink"
    ceg.nodes[ceg.sink_node][max_dist] = 0
    node_queue = [ceg.sink_node]

    while node_queue != [ceg.root_node]:
        node = node_queue.pop(0)
        for pred in ceg.predecessors(node):
            max_dist_to_sink = set()
            for succ in ceg.successors(pred):
                try:
                    max_dist_to_sink.add(
                        ceg.nodes[succ][max_dist]
                    )
                    ceg.nodes[pred][max_dist] = max(max_dist_to_sink) + 1
                except KeyError:
                    break

            if pred not in node_queue:
                node_queue.append(pred)


def _gen_nodes_with_increasing_distance(ceg: ChainEventGraph, start=0) -> list:
    """Generates nodes that are either the same or further
    from the sink node than the last node generated."""
    max_dists = nx.get_node_attributes(ceg, 'max_dist_to_sink')
    distance_dict: Mapping[int, Iterable[str]] = {}
    for node, distance in max_dists.items():
        dist_list: List = distance_dict.setdefault(distance, [])
        dist_list.append(node)

    for dist in range(0, max(distance_dict) + 1):
        nodes = distance_dict.get(dist)
        if dist >= start and nodes is not None:
            yield nodes


def _check_nodes_can_be_merged(ceg: ChainEventGraph, node_1, node_2) -> bool:
    """Determine if the two nodes are able to be merged."""
    have_same_successor_nodes = (
        set(ceg.adj[node_1].keys()) == set(ceg.adj[node_2].keys())
    )

    if have_same_successor_nodes:
        have_same_outgoing_edges = True
        v1_adj = ceg.succ[node_1]
        for succ_node in list(v1_adj.keys()):
            v1_edges = ceg.succ[node_1][succ_node]
            v2_edges = ceg.succ[node_2][succ_node]

            if v1_edges is None or v2_edges is None:
                have_same_outgoing_edges &= False
                break

            v2_edge_labels = list(v2_edges.keys())

            for label in v1_edges.keys():
                if label not in v2_edge_labels:
                    have_same_outgoing_edges &= False
                    break
                have_same_outgoing_edges &= True
    else:
        have_same_outgoing_edges = False

    try:
        in_same_stage = (
            ceg.nodes[node_1]['stage'] == ceg.nodes[node_2]['stage']
        )
    except KeyError:
        in_same_stage = False

    return in_same_stage and (
        have_same_successor_nodes and have_same_outgoing_edges
    )
