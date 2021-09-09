
import pydotplus as pdp
import networkx as nx
from copy import deepcopy
import itertools as it
import collections

from ..utilities.util import Util
from IPython.display import Image
from IPython import get_ipython
import logging

logger = logging.getLogger('pyceg.chain_event_graph')


class ChainEventGraph(nx.MultiDiGraph):
    """
    Class: Chain Event Graph

    Input: Staged tree object (StagedTree)
    Output: Chain event graphs
    """
    def __init__(self, incoming_graph_data,
                 node_prefix='w', sink_suffix='inf', **attr):
        super().__init__(incoming_graph_data, **attr)
        self.sink_suffix = sink_suffix
        self.node_prefix = node_prefix
        # rename root node:
        nx.relabel_nodes(self, {'s0': self.root_node}, copy=False)
        self.__trim_leaves_from_graph()
        self.ahc_output = deepcopy(incoming_graph_data.ahc_output)

        if self.ahc_output == {}:
            raise ValueError("Run staged tree AHC transitions first.")
        self.certain_evidence = Evidence()
        self.uncertain_evidence = Evidence()

    @property
    def node_prefix(self):
        return self._node_prefix

    @node_prefix.setter
    def node_prefix(self, value):
        self._node_prefix = str(value)

    @property
    def sink_suffix(self):
        return self._sink_suffix

    @sink_suffix.setter
    def sink_suffix(self, value):
        self._sink_suffix = str(value)

    @property
    def sink_node(self):
        return "%s%s" % (self.node_prefix, self.sink_suffix)

    @property
    def root_node(self):
        return ("%s0" % self.node_prefix)

    @property
    def certain_evidence(self):
        return self._certain_evidence

    @certain_evidence.setter
    def certain_evidence(self, value):
        self._certain_evidence = value

    @property
    def uncertain_evidence(self):
        return self._uncertain_evidence

    @uncertain_evidence.setter
    def uncertain_evidence(self, value):
        self._uncertain_evidence = value

    @property
    def evidence_as_str(self) -> str:
        def evidence_str(base_str, evidence):
            if evidence.edges == []:
                base_str += '   Edges = []\n'
            else:
                base_str += '   Edges = [\n'
                for edge in evidence.edges:
                    base_str += '     %s,\n' % (str(edge))
                base_str += '   ]\n'

            if evidence.vertices == set():
                base_str += '   Vertices = {}\n'
            else:
                base_str += '   Vertices = {\n'
                for vertex in evidence.vertices:
                    base_str += "     '%s',\n" % (str(vertex))
                base_str += '   }\n\n'
            return base_str

        base_str = 'The evidence you have given is as follows:\n'
        base_str += ' Evidence you are certain of:\n'
        base_str = evidence_str(base_str, self.certain_evidence)
        base_str += ' Evidence you are uncertain of:\n'
        base_str = evidence_str(base_str, self.uncertain_evidence)
        return base_str

    def clear_evidence(self):
        self.certain_evidence = Evidence()
        self.uncertain_evidence = Evidence()

    def _check_evidence_consistency(self, type_of_evidence,
                                    evidence, certain) -> bool:
        pass

    def _find_edges_entering_or_exiting_node(
            self, node, direction='in') -> list:

        """When a node is provided, this functions finds all
        edges either coming in or going out and returns them."""
        node_data = self.graph['nodes'][node]

        if direction not in ['in', 'out']:
            raise(
                ValueError(
                    "Invalid direction provided, please specify" +
                    " either 'in', or 'out'"
                )
            )
        elif (direction == 'out' and node_data['sink']) or \
             (direction == 'in' and node_data['root']):
            return []
        else:
            node_edge_key = direction + 'going_edges'
            edge_keys = node_data[node_edge_key]
            edges = []
            for edge_key in edge_keys:
                for edge in self.graph['edges'][edge_key]:
                    edges.append(edge)
            return edges

    def _extend_paths_to_root(self, paths) -> list:
        """Takes a set of paths, and steps through the graph
        to the root node to find all the paths."""
        root_not_found = True

        while root_not_found:
            extended_paths = []
            for path in paths:
                parent = path[0][Edge.SRC.value]

                if self.graph['nodes'][parent]['root']:
                    root_not_found = False
                    break
                else:
                    root_not_found = True
                edges = self._find_edges_entering_or_exiting_node(parent, 'in')
                for edge in edges:
                    new_edge = (edge['src'], edge['dest'], edge['label'])
                    extended_paths.append([new_edge] + path)

            if root_not_found:
                paths = extended_paths

        return paths

    def _extend_paths_to_sink(self, paths) -> list:
        """Takes a set of paths, and steps through the graph
        to the sink node to find all the paths."""
        sink_not_found = True
        while sink_not_found:
            extended_paths = []
            for path in paths:
                child = path[-1][Edge.DST.value]

                if self.graph['nodes'][child]['sink'] or\
                        self.graph['nodes'][child]['outgoing_edges'] == []:
                    sink_not_found = False
                    break
                else:
                    edges = self._find_edges_entering_or_exiting_node(
                        child, 'out'
                    )
                    sink_not_found = True
                    for edge in edges:
                        new_edge = (edge['src'], edge['dest'], edge['label'])
                        extended_paths.append(path + [new_edge])

            if sink_not_found:
                paths = extended_paths

        return paths

    def _find_paths_containing_edge(self, edge) -> set:
        """When provided with an edge, produces all the paths
        that pass through that edge."""
        # Edge has format ('src', 'dest', 'label')
        paths = [[edge]]
        paths = self._extend_paths_to_sink(paths)
        paths = self._extend_paths_to_root(paths)

        return set(map(frozenset, paths))

    def _find_paths_containing_node(self, node) -> set:
        """When provided with a node in the graph,
        provides all the paths that pass through the node"""
        paths = []
        node_data = self.graph['nodes'][node]
        edge_keys = node_data['ingoing_edges'] if node_data['sink'] \
            else node_data['outgoing_edges']

        for key in edge_keys:
            edges = self.graph['edges'][key]
            for edge_data in edges:
                edge = (
                    edge_data['src'],
                    edge_data['dest'],
                    edge_data['label']
                )

                paths.append([edge])

        if not node_data['sink']:
            paths = self._extend_paths_to_sink(paths)

        if not node_data['root']:
            paths = self._extend_paths_to_root(paths)

        return set(map(frozenset, paths))

    def __update_distances_of_nodes_to_sink_node(self) -> None:
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
                    max_dist_to_sink.add(
                        self.nodes[succ][max_dist]
                    )

                self.nodes[pred][max_dist] = max(max_dist_to_sink) + 1
                if pred not in node_queue:
                    node_queue.append(pred)

    def __gen_nodes_with_increasing_distance(self, start=0) -> list:
        max_dists = nx.get_node_attributes(self, 'max_dist_to_sink')

        distance_dict = {}
        for key, value in max_dists.items():
            distance_dict.setdefault(value, []).append(key)

        for dist in range(len(distance_dict)):
            if dist >= start:
                yield distance_dict[dist]

    def __get_next_node_name(self):
        try:
            num = str(next(self._num_iter))
        except AttributeError:
            self._num_iter = it.count(1, 1)
            num = str(next(self._num_iter))

        return str(self.node_prefix) + num

    def __trim_leaves_from_graph(self):
        # Create new CEG sink node
        self.add_node(self.sink_node, colour='lightgrey')
        outgoing_edges = deepcopy(self.succ).items()
        # Check to see if any nodes have no outgoing edges.
        for node, outgoing_edges in outgoing_edges:
            if outgoing_edges == {} and node != self.sink_node:
                incoming_edges = deepcopy(self.pred[node]).items()
                # When node is identified as a leaf check the
                # predessesor nodes that have edges that enter this node.
                for pred_node, edges in incoming_edges:
                    for _, edge in edges.items():
                        # Create new edge that points to the sink node,
                        # with all the same data as the edge we will delete.
                        self.add_edge(
                            pred_node,
                            self.sink_node,
                            key=edge['label'],
                            label=edge['label'],
                            count=edge['count'],
                            prior=edge['prior'],
                            posterior=edge['posterior']
                        )
                self.remove_node(node)

    def generate_CEG(self):
        '''
        This function takes the output of the AHC algorithm and identifies
        the positions i.e. the vertices of the CEG and the edges of the CEG
        along with their edge labels and edge counts. Here we use the
        algorithm in our paper with the optimal stopping time.
        '''
        def check_vertices_can_be_merged(v1, v2) -> bool:
            has_same_successor_nodes = \
                set(self.adj[v1].keys()) == set(self.adj[v2].keys())

            has_same_outgoing_edges = True
            v1_adj = self.succ[v1]
            for succ_node in list(v1_adj.keys()):
                v1_edges = self.get_edge_data(v1, succ_node)
                v2_edges = self.get_edge_data(v2, succ_node)

                if v1_edges is None or v2_edges is None:
                    has_same_outgoing_edges &= False
                    break

                v2_edge_labels = \
                    [edge['label'] for _, edge in v2_edges.items()]

                for v1_edge in v1_edges:
                    v1_edge_label = v1_edges[v1_edge]['label']
                    if v1_edge_label not in v2_edge_labels:
                        has_same_outgoing_edges &= False
                        break
                    else:
                        has_same_outgoing_edges &= True

            try:
                in_same_stage = \
                    self.nodes[v1]['stage'] == self.nodes[v2]['stage']
            except KeyError:
                in_same_stage = False

            return in_same_stage and \
                has_same_successor_nodes and has_same_outgoing_edges

        def merge_outgoing_edges(nodes):
            for node in list(nodes):
                for succ, edge_dict in self.succ[node].items():
                    edges = list(edge_dict.keys())

                    while edges != []:
                        edge = edges.pop(0)
                        label = edge_dict[edge]['label']
                        other_edges = edges[:]
                        while other_edges != []:
                            other_edge = other_edges.pop(0)
                            other_label = edge_dict[other_edge]['label']
                            if label == other_label:
                                edge_dict[edge]['prior'] += \
                                    edge_dict[other_edge]['prior']
                                edge_dict[edge]['count'] += \
                                    edge_dict[other_edge]['count']
                                edge_dict[edge]['posterior'] += \
                                    edge_dict[other_edge]['posterior']
                                self.remove_edge(node, succ, other_edge)
                                edges.remove(other_edge)

        def relabel_nodes(base_nodes, renamed_nodes=[]):
            next_level = []
            # first, relabel the successors of this node
            for node in base_nodes:
                node_mapping = {}
                for succ in self.succ[node].keys():
                    if succ != self.sink_node and succ not in renamed_nodes:
                        node_mapping[succ] = self.__get_next_node_name()
                        next_level.append(node_mapping[succ])
                        renamed_nodes.append(node_mapping[succ])

                if node_mapping != {}:
                    nx.relabel_nodes(
                        self,
                        node_mapping,
                        copy=False
                    )
            if next_level != []:
                relabel_nodes(next_level, renamed_nodes)

        self.__update_distances_of_nodes_to_sink_node()
        src_node_gen = self.__gen_nodes_with_increasing_distance(
            start=1
        )
        next_set_of_nodes = next(src_node_gen)

        while next_set_of_nodes != [self.root_node]:
            merged_nodes = set()
            while next_set_of_nodes != []:
                node_1 = next_set_of_nodes.pop(0)
                for node_2 in next_set_of_nodes:
                    mergeable = check_vertices_can_be_merged(node_1, node_2)
                    if mergeable:
                        nx.relabel_nodes(self, {node_2: node_1}, copy=False)
                        next_set_of_nodes.remove(node_2)
                        merged_nodes.add(node_1)

            merge_outgoing_edges(merged_nodes)

            try:
                next_set_of_nodes = next(src_node_gen)
            except StopIteration:
                next_set_of_nodes = []

        relabel_nodes([self.root_node])

    def create_figure(self, filename):
        filename, filetype = Util.generate_filename_and_mkdir(filename)

        graph = pdp.Dot(graph_type='digraph', rankdir='LR')

        edge_labels = nx.get_edge_attributes(self, 'label')
        edge_probabilities = nx.get_edge_attributes(self, 'posterior')

        for edge, label in edge_labels.items():
            full_label = label + '\n' + str(edge_probabilities[edge])
            graph.add_edge(
                pdp.Edge(
                    src=edge[0],
                    dst=edge[1],
                    label=full_label,
                    labelfontcolor='#009933',
                    fontsize='10.0',
                    color='black'
                )
            )

        for node, node_data in self.nodes.items():
            fill_colour = node_data['colour']

            graph.add_node(
                pdp.Node(
                    name=node,
                    label=node,
                    style='filled',
                    fillcolor=fill_colour
                )
            )

        graph.write(str(filename), format=filetype)

        if get_ipython() is None:
            return None
        else:
            return Image(graph.create_png())


class Evidence:
    Edge = collections.namedtuple('edge', ['u', 'v', 'label'])

    def __init__(self):
        self.edges = []
        self.vertices = set()

    @property
    def edges(self):
        return self.__edges

    @edges.setter
    def edges(self, value):
        self.__edges = value

    @property
    def vertices(self):
        return self.__vertices

    @vertices.setter
    def vertices(self, value):
        self.__vertices = value

    def __repr__(self) -> str:
        return "Evidence(Edges=%s, Vertices=%s)" %\
            (str(self.edges), str(self.vertices))

    def add_edge(self, u, v, label):
        self.edges.append(Evidence.Edge(u, v, label))

    def remove_edge(self, u, v, label):
        self.edges.remove(Evidence.Edge(u, v, label))

    def add_vertex(self, vertex):
        self.vertices.add(vertex)

    def remove_vertex(self, vertex):
        self.vertices.remove(vertex)
