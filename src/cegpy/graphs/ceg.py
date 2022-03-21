"""Chain Event Graph"""

from copy import deepcopy
import itertools as it
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
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
    path_list: List
    _stages: Mapping[str, List[str]]
    _node_num_iterator: it.count

    def __init__(self, staged_tree: Optional[StagedTree] = None, **attr):
        self.ahc_output = deepcopy(getattr(staged_tree, "ahc_output", None))
        super().__init__(staged_tree, **attr)
        self.sink_suffix = "&infin;"
        self.node_prefix = "w"
        self._stages = {}
        self._path_list = []
        self._node_num_iterator = it.count(1, 1)

    @property
    def sink_node(self) -> str:
        """Sink node name as a string."""
        return f"{self.node_prefix}{self.sink_suffix}"

    @property
    def root_node(self) -> str:
        """Root node name as a string."""
        return f"{self.node_prefix}0"

    @property
    def stages(self) -> Mapping[str, List[str]]:
        """Mapping of stages to constituent nodes."""
        node_stages = dict(self.nodes(data='stage', default=None))
        for node, stage in node_stages.items():
            try:
                self._stages[stage].append(node)
            except KeyError:
                self._stages[stage] = [node]

        return self._stages

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
        self._update_probabilities()
        self._trim_leaves_from_graph()
        self._update_distances_of_nodes_to_sink_node()
        src_node_gen = self._gen_nodes_with_increasing_distance(
            start=1
        )
        next_set_of_nodes = next(src_node_gen)

        while next_set_of_nodes != [self.root_node]:
            nodes_to_merge = set()
            while len(next_set_of_nodes) > 1:
                node_1 = next_set_of_nodes.pop(0)
                for node_2 in next_set_of_nodes:
                    mergeable = self._check_nodes_can_be_merged(
                        node_1, node_2
                    )
                    if mergeable:
                        nodes_to_merge.add((node_1, node_2))

            self._merge_nodes(nodes_to_merge)

            try:
                next_set_of_nodes = next(src_node_gen)
            except StopIteration:
                next_set_of_nodes = []

        self._relabel_nodes([self.root_node])
        self._update_path_list()

    def _relabel_nodes(self, base_nodes, renamed_nodes=[]):
        next_level = []
        # first, relabel the successors of this node
        for node in base_nodes:
            node_mapping = {}
            for succ in self.succ[node].keys():
                if succ != self.sink_node and succ not in renamed_nodes:
                    node_mapping[succ] = self._get_next_node_name()
                    next_level.append(node_mapping[succ])
                    renamed_nodes.append(node_mapping[succ])

            if node_mapping:
                nx.relabel_nodes(
                    self,
                    node_mapping,
                    copy=False
                )
        if next_level:
            self._relabel_nodes(next_level, renamed_nodes)

    def _merge_nodes(self, nodes_to_merge):
        """nodes to merge should be a set of 2 element tuples"""
        temp_1 = 'temp_1'
        temp_2 = 'temp_2'
        while nodes_to_merge != set():
            nodes = nodes_to_merge.pop()
            new_node = nodes[0]
            # Copy nodes to temp nodes
            node_map = {
                nodes[0]: temp_1,
                nodes[1]: temp_2
            }
            nx.relabel_nodes(self, node_map, copy=False)
            ebunch_to_remove = []  # List of edges to remove
            self.add_node(new_node)
            for succ, t1_edge_dict in self.succ[temp_1].items():
                edge_labels = list(t1_edge_dict.keys())
                while edge_labels:
                    label = edge_labels.pop(0)
                    t1_edge = t1_edge_dict[label]
                    t2_edge = self.succ[temp_2][succ][label]

                    new_edge_data = _merge_edge_data(
                        edge_1=t1_edge,
                        edge_2=t2_edge,
                    )

                    try:
                        new_edge_data['probability'] = (
                            t1_edge['probability']
                        )
                        self.add_edge(
                            u_for_edge=new_node,
                            v_for_edge=succ,
                            key=label,
                            count=new_edge_data['count'],
                            prior=new_edge_data['prior'],
                            posterior=new_edge_data['posterior'],
                            probability=new_edge_data['probability']
                        )
                    except KeyError:
                        self.add_edge(
                            u_for_edge=new_node,
                            v_for_edge=succ,
                            key=label,
                            count=new_edge_data['count'],
                            prior=new_edge_data['prior'],
                            posterior=new_edge_data['posterior']
                        )
                    ebunch_to_remove.append((temp_1, succ, label))
                    ebunch_to_remove.append((temp_2, succ, label))

            self.remove_edges_from(ebunch_to_remove)
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

    def _check_nodes_can_be_merged(self, node_1, node_2) -> bool:
        """Determine if the two nodes are able to be merged."""
        has_same_successor_nodes = \
            set(self.adj[node_1].keys()) == set(self.adj[node_2].keys())

        if has_same_successor_nodes:
            has_same_outgoing_edges = True
            v1_adj = self.succ[node_1]
            for succ_node in list(v1_adj.keys()):
                v1_edges = self.succ[node_1][succ_node]
                v2_edges = self.succ[node_2][succ_node]

                if v1_edges is None or v2_edges is None:
                    has_same_outgoing_edges &= False
                    break

                v2_edge_labels = list(v2_edges.keys())

                for label in v1_edges.keys():
                    if label not in v2_edge_labels:
                        has_same_outgoing_edges &= False
                        break
                    has_same_outgoing_edges &= True
        else:
            has_same_outgoing_edges = False

        try:
            in_same_stage = \
                self.nodes[node_1]['stage'] == self.nodes[node_2]['stage']
        except KeyError:
            in_same_stage = False

        return in_same_stage and \
            has_same_successor_nodes and has_same_outgoing_edges

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
            full_label = f"{label}\n{probability:.2f}"
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

    def _update_probabilities(self):
        count_total_lbl = 'count_total'
        edge_counts = list(self.edges(data='count', keys=True, default=0))

        for stage, stage_nodes in self.stages.items():
            count_total = 0
            stage_edges = {}
            if stage is not None:

                for (src, _, label, count) in edge_counts:
                    if src in stage_nodes:
                        count_total += count
                        try:
                            stage_edges[label] += count
                        except KeyError:
                            stage_edges[label] = count

                for node in stage_nodes:
                    self.nodes[node][count_total_lbl] = count_total

                for (src, dst, label, _) in edge_counts:
                    if src in stage_nodes:
                        self.edges[src, dst, label]['probability'] = (
                            stage_edges[label] / count_total
                        )
            else:
                for node in stage_nodes:
                    count_total = 0
                    stage_edges = {}
                    for (src, _, label, count) in edge_counts:
                        if src == node:
                            count_total += count
                            try:
                                stage_edges[label] += count
                            except KeyError:
                                stage_edges[label] = count

                    self.nodes[node][count_total_lbl] = count_total
                    for (src, dst, label, _) in edge_counts:
                        if src == node:
                            self.edges[src, dst, label]['probability'] = (
                                stage_edges[label] / count_total
                            )

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
                self._path_list = path_list
                break

    def _update_distances_of_nodes_to_sink_node(self) -> None:
        """
        Iterates through the graph until it finds the root node.
        For each node, it determines the maximum number of edges
        from that node to the sink node.
        """
        max_dist = 'max_dist_to_sink'
        self.nodes[self.sink_node][max_dist] = 0
        node_queue = [self.sink_node]

        while node_queue != [self.root_node]:
            node = node_queue.pop(0)
            for pred in self.predecessors(node):
                max_dist_to_sink = set()
                for succ in self.successors(pred):
                    try:
                        max_dist_to_sink.add(
                            self.nodes[succ][max_dist]
                        )
                        self.nodes[pred][max_dist] = max(max_dist_to_sink) + 1
                    except KeyError:
                        break

                if pred not in node_queue:
                    node_queue.append(pred)

    def _gen_nodes_with_increasing_distance(self, start=0) -> list:
        """Generates nodes that are either the same or further
        from the sink node than the last node generated."""
        max_dists = nx.get_node_attributes(self, 'max_dist_to_sink')
        distance_dict: Mapping[int, Iterable[str]] = {}
        for node, distance in max_dists.items():
            dist_list: List = distance_dict.setdefault(distance, [])
            dist_list.append(node)

        for node_idx, dist in enumerate(distance_dict):
            if dist >= start:
                yield distance_dict[node_idx]

    def _get_next_node_name(self):
        """Generates sequentially increasing node numbers."""
        return f"{self.node_prefix}{next(self._node_num_iterator)}"

    def _trim_leaves_from_graph(self):
        """Trims all the leaves from the graph, and points each incoming
        edge to the sink node."""
        # Create new CEG sink node
        self.add_node(self.sink_node, colour='lightgrey')
        outgoing_edges = deepcopy(self.succ).items()
        # Check to see if any nodes have no outgoing edges.
        for node, out_edges in outgoing_edges:
            if not out_edges and node != self.sink_node:
                all_inc_edges: Mapping[str, Dict[str, Dict]] = (
                    dict(deepcopy(self.pred[node]))
                )
                # When node is identified as a leaf check the
                # predessesor nodes that have edges that enter this node.
                for pred_node, inc_edges_to_node in all_inc_edges.items():
                    for edge_label, edge_data in inc_edges_to_node.items():
                        # Create new edge that points to the sink node,
                        # with all the same data as the edge we will delete.
                        prob = edge_data.get("probability", 1)
                        self.add_edge(
                            pred_node,
                            self.sink_node,
                            key=edge_label,
                            count=edge_data['count'],
                            prior=edge_data['prior'],
                            posterior=edge_data['posterior'],
                            probability=prob
                        )
                self.remove_node(node)


def _merge_edge_data(
    edge_1: Dict[str, Any],
    edge_2: Dict[str, Any],
) -> Dict[str, Any]:
    """Merges the counts, priors, and posteriors of two edges."""
    new_edge_data = {}
    for key in edge_1:
        new_edge_data[key] = edge_1[key] + edge_2[key]
    return new_edge_data


def _relabel_nodes(ceg: ChainEventGraph):
    nodes_to_rename = list(ceg.succ[ceg.root_node].keys())
    # first, relabel the successors of this node
    node_mapping = {}
    while nodes_to_rename:
        for node in nodes_to_rename.copy():
            node_mapping[node] = ceg._get_next_node_name()
            for succ in ceg.succ[node].keys():
                if (succ != ceg.sink_node and succ not in nodes_to_rename):
                    nodes_to_rename.append(succ)
            nodes_to_rename.remove(node)

    nx.relabel_nodes(
        ceg,
        node_mapping,
        copy=False
    )
