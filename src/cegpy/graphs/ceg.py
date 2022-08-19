import pydotplus as pdp
import networkx as nx
from copy import deepcopy
import itertools as it

from ..utilities.util import Util
from IPython.display import Image
from IPython import get_ipython
import logging

logger = logging.getLogger('cegpy.chain_event_graph')


class ChainEventGraph(nx.MultiDiGraph):
    """
    Class: Chain Event Graph

    Input: Staged tree object (StagedTree)
    Output: Chain event graphs
    """
    def __init__(self, staged_tree=None, **attr):
        super().__init__(staged_tree, **attr)
        self.sink_suffix = '&infin;'
        self.node_prefix = 'w'

        if staged_tree is not None:
            try:
                self.ahc_output = deepcopy(staged_tree.ahc_output)
            except AttributeError:
                self.ahc_output = {}
        else:
            logger.info("Class called with no incoming graph.")
        self.evidence = Evidence(self)

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
    def path_list(self):
        return self.__path_list

    @path_list.setter
    def path_list(self, value):
        self.__path_list = value

    @property
    def reduced(self):
        return self.evidence.reduced_graph

    @property
    def stages(self):
        self.__stages = {}
        node_stages = dict(self.nodes(data='stage', default=None))
        for k, v in node_stages.items():
            try:
                self.__stages[v].append(k)
            except KeyError:
                self.__stages[v] = [k]

        return self.__stages

    def clear_evidence(self):
        self.evidence = Evidence(self)

    def generate(self):
        '''
        This function takes the output of the AHC algorithm and identifies
        the positions i.e. the vertices of the CEG and the edges of the CEG
        along with their edge labels and edge counts. Here we use the
        algorithm in our paper with the optimal stopping time.
        '''
        def check_vertices_can_be_merged(v1, v2) -> bool:
            has_same_successor_nodes = \
                set(self.adj[v1].keys()) == set(self.adj[v2].keys())

            if has_same_successor_nodes:
                has_same_outgoing_edges = True
                v1_adj = self.succ[v1]
                for succ_node in list(v1_adj.keys()):
                    v1_edges = self.succ[v1][succ_node]
                    v2_edges = self.succ[v2][succ_node]

                    if v1_edges is None or v2_edges is None:
                        has_same_outgoing_edges &= False
                        break

                    v2_edge_labels = \
                        [label for label in v2_edges.keys()]

                    for label in v1_edges.keys():
                        if label not in v2_edge_labels:
                            has_same_outgoing_edges &= False
                            break
                        else:
                            has_same_outgoing_edges &= True
            else:
                has_same_outgoing_edges = False

            try:
                in_same_stage = \
                    self.nodes[v1]['stage'] == self.nodes[v2]['stage']
            except KeyError:
                in_same_stage = False

            return in_same_stage and \
                has_same_successor_nodes and has_same_outgoing_edges

        def merge_nodes(nodes_to_merge):
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
                    while edge_labels != []:
                        new_edge_data = {}
                        label = edge_labels.pop(0)
                        t1_edge = t1_edge_dict[label]
                        t2_edge = self.succ[temp_2][succ][label]
                        new_edge_data['count'] = \
                            t1_edge['count'] + t2_edge['count']
                        new_edge_data['prior'] = \
                            t1_edge['prior'] + t2_edge['prior']
                        new_edge_data['posterior'] = \
                            t1_edge['posterior'] + t2_edge['posterior']
                        try:

                            new_edge_data['probability'] = \
                                t1_edge['probability']
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
                pass

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

        if self.ahc_output == {}:
            raise ValueError("Run staged tree AHC transitions first.")
        # rename root node:
        nx.relabel_nodes(self, {'s0': self.root_node}, copy=False)
        self.__update_probabilities()
        self.__trim_leaves_from_graph()
        self.__update_distances_of_nodes_to_sink_node()
        src_node_gen = self.__gen_nodes_with_increasing_distance(
            start=1
        )
        next_set_of_nodes = next(src_node_gen)

        while next_set_of_nodes != [self.root_node]:
            nodes_to_merge = set()
            while len(next_set_of_nodes) > 1:
                node_1 = next_set_of_nodes.pop(0)
                for node_2 in next_set_of_nodes:
                    mergeable = check_vertices_can_be_merged(node_1, node_2)
                    if mergeable:
                        nodes_to_merge.add((node_1, node_2))

            merge_nodes(nodes_to_merge)

            try:
                next_set_of_nodes = next(src_node_gen)
            except StopIteration:
                next_set_of_nodes = []

        relabel_nodes([self.root_node])
        self.__update_path_list()

    @property
    def dot_graph(self):
        return self._generate_dot_graph()

    def _generate_dot_graph(self):
        graph = pdp.Dot(graph_type='digraph', rankdir='LR')
        edge_probabilities = list(
            self.edges(data='probability', default=1, keys=True)
        )

        for (u, v, k, p) in edge_probabilities:
            full_label = "{}\n{:.2f}".format(k, p)
            graph.add_edge(
                pdp.Edge(
                    src=u,
                    dst=v,
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

    def create_figure(self, filename):
        """
        Draws the chain event graph representation of the stage tree,
        and saves it to "<filename>.filetype". Supports any filetype that
        graphviz supports. e.g: "event_tree.png" or "event_tree.svg" etc.
        """
        filename, filetype = Util.generate_filename_and_mkdir(filename)
        graph = self.dot_graph
        graph.write(str(filename), format=filetype)

        if get_ipython() is None:
            return None
        else:
            return Image(graph.create_png())

    def __update_probabilities(self):
        count_total_lbl = 'count_total'
        edge_counts = list(self.edges(data='count', keys=True, default=0))

        for stage, stage_nodes in self.stages.items():
            count_total = 0
            stage_edges = {}
            if stage is not None:

                for (u, _, k, c) in edge_counts:
                    if u in stage_nodes:
                        count_total += c
                        try:
                            stage_edges[k] += c
                        except KeyError:
                            stage_edges[k] = c

                for node in stage_nodes:
                    self.nodes[node][count_total_lbl] = count_total

                for (u, v, k, _) in edge_counts:
                    if u in stage_nodes:
                        self.edges[u, v, k]['probability'] =\
                            stage_edges[k] / count_total
            else:
                for node in stage_nodes:
                    count_total = 0
                    stage_edges = {}
                    for (u, _, k, c) in edge_counts:
                        if u == node:
                            count_total += c
                            try:
                                stage_edges[k] += c
                            except KeyError:
                                stage_edges[k] = c

                    self.nodes[node][count_total_lbl] = count_total
                    for (u, v, k, _) in edge_counts:
                        if u == node:
                            self.edges[u, v, k]['probability'] =\
                                stage_edges[k] / count_total

    def __update_path_list(self) -> None:
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
                    try:
                        max_dist_to_sink.add(
                            self.nodes[succ][max_dist]
                        )
                        self.nodes[pred][max_dist] = max(max_dist_to_sink) + 1
                    except KeyError:
                        break

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
                    for edge_label, edge in edges.items():
                        # Create new edge that points to the sink node,
                        # with all the same data as the edge we will delete.
                        try:
                            prob = edge['probability']
                        except KeyError:
                            prob = 1
                        self.add_edge(
                            pred_node,
                            self.sink_node,
                            key=edge_label,
                            count=edge['count'],
                            prior=edge['prior'],
                            posterior=edge['posterior'],
                            probability=prob
                        )
                self.remove_node(node)


class Evidence:
    CERTAIN = True
    UNCERTAIN = False

    def __init__(self, graph):
        self.__graph = graph

        self.certain_edges = []
        self.uncertain_edges = []
        self.certain_vertices = set()
        self.uncertain_vertices = set()

    @property
    def reduced_graph(self):
        return self.__create_reduced_graph()

    @property
    def path_list(self):
        return self._path_list

    @path_list.setter
    def path_list(self, value):
        self._path_list = value

    @property
    def edges(self):
        return list(self._edges)

    @edges.setter
    def edges(self, value):
        self._edges = value

    @property
    def vertices(self):
        return list(self._vertices)

    @vertices.setter
    def vertices(self, value):
        self._vertices = value

    def add_edge(self, u, v, label, certain):
        edge = (u, v, label)
        if certain:
            self.certain_edges.append(edge)
        else:
            self.uncertain_edges.append(edge)

    def add_edges_from(self, edges, certain):
        for (u, v, k) in edges:
            self.add_edge(u, v, k, certain)

    def remove_edge(self, u, v, label, certain):
        if certain:
            self.certain_edges.remove((u, v, label))
        else:
            self.uncertain_edges.remove((u, v, label))

    def remove_edges_from(self, edges, certain):
        for (u, v, k) in edges:
            self.remove_edge(u, v, k, certain)

    def add_node(self, node, certain):
        if certain:
            self.certain_vertices.add(node)
        else:
            self.uncertain_vertices.add(node)

    def add_nodes_from(self, nodes, certain):
        for node in nodes:
            self.add_node(node, certain)

    def remove_node(self, node, certain):
        if certain:
            self.certain_vertices.remove(node)
        else:
            self.uncertain_vertices.remove(node)

    def remove_nodes_from(self, nodes, certain):
        for node in nodes:
            self.remove_node(node, certain)

    def __repr__(self) -> str:
        repr = "Evidence(CertainEdges={}, CertainVertices={}," +\
            " UncertainEdges={}, UncertainVertices={})"
        return repr.format(
            str(self.certain_edges),
            str(self.certain_vertices),
            str(self.uncertain_edges),
            str(self.uncertain_vertices)
        )

    def __str__(self) -> str:
        """Returns human readable version of the evidence you've provided."""
        def evidence_str(base_str, edges, vertices):
            if edges == []:
                base_str += '   Edges = []\n'
            else:
                base_str += '   Edges = [\n'
                for edge in edges:
                    base_str += '     %s,\n' % (str(edge))
                base_str += '   ]\n'

            if vertices == set():
                base_str += '   Vertices = {}\n'
            else:
                base_str += '   Vertices = {\n'
                for vertex in vertices:
                    base_str += "     '%s',\n" % (str(vertex))
                base_str += '   }\n\n'
            return base_str

        base_str = 'The evidence you have given is as follows:\n'
        base_str += ' Evidence you are certain of:\n'
        base_str = evidence_str(
            base_str,
            self.certain_edges,
            self.certain_vertices
        )
        base_str += ' Evidence you are uncertain of:\n'
        base_str = evidence_str(
            base_str,
            self.uncertain_edges,
            self.uncertain_vertices
        )
        return base_str

    def __update_path_list(self):
        def remove_paths(paths_list, paths_to_remove):
            for path in paths_to_remove:
                paths_list.remove(path)

        # Certain Evidence
        certain_path_list = self.__graph.path_list.copy()
        # Uncertain Evidence
        uncertain_path_list = self.__graph.path_list.copy()

        if self.certain_edges or self.certain_vertices:
            # Apply certain Edges
            paths_to_remove = []
            for edge in self.certain_edges:
                for path in certain_path_list:
                    if edge not in path:
                        paths_to_remove.append(path)

            remove_paths(certain_path_list, paths_to_remove)

            # Apply certain vertices
            paths_to_remove = []
            for vertex in self.certain_vertices:
                for path in certain_path_list:
                    vertex_in_path = False
                    for (u, v, _) in path:
                        if vertex == u or vertex == v:
                            vertex_in_path = True
                            break

                    if not vertex_in_path:
                        paths_to_remove.append(path)

            remove_paths(certain_path_list, paths_to_remove)

        if self.uncertain_edges or self.uncertain_vertices:
            # Apply uncertain Edges
            paths_to_include = []
            for edge in self.uncertain_edges:
                for path in uncertain_path_list:
                    if edge in path:
                        paths_to_include.append(path)

            # Apply uncertain Vertices
            for vertex in self.uncertain_vertices:
                for path in uncertain_path_list:
                    vertex_in_path = False
                    for (u, v, _) in path:
                        if vertex == u or vertex == v:
                            vertex_in_path = True
                            break

                    if vertex_in_path:
                        paths_to_include.append(path)

            uncertain_path_list = paths_to_include

        # Take paths that are found in both certain and uncertain lists
        if not self.uncertain_edges and not self.uncertain_vertices:
            self.path_list = certain_path_list
        elif not self.certain_edges and not self.certain_vertices:
            self.path_list = uncertain_path_list
        else:
            new_path_list = []
            for path in certain_path_list:
                if path in uncertain_path_list:
                    new_path_list.append(path)
            self.path_list = new_path_list

    def __update_edges_and_vertices(self):
        edges = set()
        vertices = set()

        for path in self.path_list:
            for (u, v, k) in path:
                edges.add((u, v, k))
                vertices.add(u)
                vertices.add(v)

        self.edges = edges
        self.vertices = vertices

    def __filter_edge(self, u, v, k) -> bool:
        if (u, v, k) in self.edges:
            return True
        else:
            return False

    def __filter_node(self, n) -> bool:
        if n in self.vertices:
            return True
        else:
            return False

    def __propagate_reduced_graph_probabilities(self, graph) -> None:
        sink = graph.sink_node
        root = graph.root_node
        graph.nodes[sink]['emphasis'] = 1
        node_set = set([sink])

        while node_set != {root}:
            try:
                v_node = node_set.pop()
            except KeyError:
                break

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

    def __create_reduced_graph(self) -> ChainEventGraph:
        self.__update_path_list()
        self.__update_edges_and_vertices()
        subgraph = ChainEventGraph(
            nx.subgraph_view(
                G=self.__graph,
                filter_node=self.__filter_node,
                filter_edge=self.__filter_edge
            )
        ).copy()

        self.__propagate_reduced_graph_probabilities(subgraph)

        return subgraph
