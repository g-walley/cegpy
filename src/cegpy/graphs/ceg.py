# from ..trees.staged import StagedTree
# import pydotplus as pdp
# from operator import add


class ChainEventGraph(object):
    """
    Class: Chain Event Graph


    Input: Staged tree object (StagedTree)
    Output: Chain event graphs
    """
    def __init__(self, staged_tree=None, root=None, sink='w_inf') -> None:
        self.root = root
        self.sink = sink
        # self.st = StagedTree()
        self.st = staged_tree
        self.ahc_output = self.st.get_AHC_output().copy()

        if self.ahc_output == {}:
            raise ValueError("Run staged tree AHC transitions first.")

        if self.root is None:
            raise(ValueError('Please input the root node!'))
            # self._identify_root_node()

        self.graph = self._create_graph_representation()

    def _create_graph_representation(self) -> dict:
        """
        This function will build a dictionary that represents
        a graph. Constructed from "event tree" representation
        from ceg.trees.event.
        The output will have the form:
        {
            'nodes': {
                'node_1': {
                    'nodes_to_merge': ['n1', 'n2', ...],
                    'colour': '<hex_colour_string>'
                },
                ...
                'node_n': {
                    ...
                }
            },
            'edges': {
                ('src', 'dest'): [
                    {
                        'src': 'n1',
                        'dest': 'n2',
                        'label': 'Edge_Label',
                        'value': 54.0
                    },
                    {
                        <any other edges between
                        the same nodes>
                    },...
                ],
                ...
            }
        }
        """
        graph = {
            "nodes": {},
            "edges": {}
        }
        event_tree = self.st.get_event_tree()
        prior = self._flatten_list_of_lists(self.st.get_prior())

        # Add all nodes, and their children, to the dictionary
        for idx, edge_key in enumerate(event_tree.keys()):
            # edge has form:
            # (('path', 'to', 'label'), ('<node_name>', '<child_name>'))
            edge = self._create_new_edge(
                src=edge_key[1][0],
                dest=edge_key[1][1],
                label=edge_key[0][-1],
                value=(event_tree[edge_key] + float(prior[idx])))

            new_edge_key = edge_key[1]
            # Add src node to graph dict:
            try:
                graph['nodes'][edge['src']]['outgoing_edges'].append(
                    new_edge_key
                )
            except KeyError:
                root = True if edge['src'] == self.root else False

                graph['nodes'][edge['src']] = \
                    self._create_new_node(
                        outgoing_edges=[new_edge_key],
                        root=root
                    )
                # graph['nodes'][edge['src']]['outgoing_edges'].append(
                #     new_edge_key
                # )
            # Add dest node
            try:
                graph['nodes'][edge['dest']]['incoming_edges'].append(
                    new_edge_key
                )
            except KeyError:
                graph['nodes'][edge['dest']] = \
                    self._create_new_node()
                graph['nodes'][edge['dest']]['incoming_edges'].append(
                    new_edge_key
                )

            # Add edge to graph dict:
            try:
                graph['edges'][new_edge_key].append(edge)
            except KeyError:
                graph['edges'][new_edge_key] = []
                graph['edges'][new_edge_key].append(edge)

        return graph

    def _update_distance_to_sink(self) -> None:
        pass

    def _identify_root_node(self, graph) -> str:
        number_of_roots = 0
        root = ''
        for node in graph['nodes']:
            node_properties = graph['nodes'][node]
            if node_properties['incoming_edges'] == []:
                root = node
                number_of_roots += 1

        if number_of_roots > 1:
            raise(ValueError('Your graph has too many roots!'))
        elif number_of_roots == 1:
            return root
        else:
            raise(ValueError('No graph root was found!'))

    def _flatten_list_of_lists(self, list_of_lists) -> list:
        flat_list = []
        for sublist in list_of_lists:
            flat_list = flat_list + sublist
        return flat_list

    def _create_new_node(self, root=False, sink=False,
                         incoming_edges=[], outgoing_edges=[],
                         dist_to_sink=-1,
                         nodes_to_merge=[], colour='lightgrey') -> dict:
        """
        Generates default format of Node dictionary
        """
        return {
            'root': root,
            'sink': sink,
            'incoming_edges': incoming_edges.copy(),
            'outgoing_edges': outgoing_edges.copy(),
            'nodes_to_merge': nodes_to_merge.copy(),
            'dist_to_sink': dist_to_sink,
            'colour': colour
        }

    def _create_new_edge(self, src='', dest='', label='', value=0.0) -> list:
        """
        Generates default format of edge dictionary.
        """
        edge = {
            'src': src,
            'dest': dest,
            'label': label,
            'value': value
        }
        return edge

    def _trim_leaves_from_graph(self, graph) -> "tuple[dict, list]":
        leaves = self.st.get_leaves()
        cut_vertices = []
        graph['nodes'][self.sink] = \
            self._create_new_node(sink=True, dist_to_sink=0)

        # Check which nodes have been identified as leaves
        edges_to_delete = []
        edges_to_add = {}
        for leaf in leaves:

            # In edge list, look for edges that terminate on this leaf
            for edge_list_key in graph['edges'].keys():
                if edge_list_key[1] == leaf:
                    new_edge_list = []
                    new_edge_list_key = (edge_list_key[0], self.sink)
                    # Each edge key may have multiple edges associate with it
                    for edge in graph['edges'][edge_list_key]:
                        new_edge = edge
                        new_edge['dest'] = self.sink
                        new_edge_list.append(new_edge)

                    # remove out of date edges from the dictionary
                    edges_to_delete.append(edge_list_key)

                    cut_vertices.append(new_edge_list_key[0])
                    # clean node dict
                    graph['nodes'][edge_list_key[0]]['outgoing_edges'].remove(
                        edge_list_key
                    )
                    try:
                        # add modified edge to the dictionary
                        edges_to_add[new_edge_list_key] = \
                            edges_to_add[new_edge_list_key] + new_edge_list
                    except KeyError:
                        edges_to_add[new_edge_list_key] = \
                            new_edge_list
                    # add edge to node dict
                    outgoing_edges = \
                        graph['nodes'][new_edge_list_key[0]]['outgoing_edges']
                    outgoing_edges.append(new_edge_list_key)

                    incoming_edges = \
                        graph['nodes'][new_edge_list_key[1]]['incoming_edges']
                    incoming_edges.append(new_edge_list_key)

                    outgoing_edges = list(set(outgoing_edges))
                    graph['nodes'][new_edge_list_key[0]]['outgoing_edges'] = \
                        outgoing_edges

                    incoming_edges = list(set(incoming_edges))
                    graph['nodes'][new_edge_list_key[1]]['incoming_edges'] = \
                        incoming_edges

            # remove leaf node from the graph
            del graph['nodes'][leaf]
        # clean up old edges
        for edge in edges_to_delete:
            del graph['edges'][edge]

        graph['edges'] = {**graph['edges'], **edges_to_add}

        return (graph, cut_vertices)

    def _merge_nodes(self) -> dict:
        return self.graph

    def _ceg_positions_edges_optimal(self):
        '''
        This function takes the output of the AHC algorithm and identifies
        the positions i.e. the vertices of the CEG and the edges of the CEG
        along with their edge labels and edge counts. Here we use the
        algorithm in our paper with the optimal stopping time.
        '''
        pass

    # def _create_pdp_graph_representation(self) -> dict:
    #     graph = pdp.Dot(graph_type='digraph', rankdir='LR')
    #     event_tree = self.st.get_event_tree()
    #     prior = self._flatten_list_of_lists(self.st.get_prior())

    #     for idx, (key, count) in enumerate(event_tree.items()):
    #         # edge_index = self.edges.index(edge)
    #         path = key[0]
    #         edge = key[1]
    #         edge_details = str(path[-1]) + '\n' + \
    #             str(count + float(prior[idx]))

    #         graph.add_edge(
    #             pdp.Edge(
    #                 edge[0],
    #                 edge[1],
    #                 label=edge_details,
    #                 labelfontcolor="#009933",
    #                 fontsize="10.0",
    #                 color="black"
    #             )
    #         )

    #     for node in self.st.get_nodes():
    #         # if colours:
    #         #     fill_colour = colours[node]
    #         # else:
    #         #     fill_colour = 'lightgrey'

    #         graph.add_node(
    #             pdp.Node(
    #                 name=node,
    #                 label=node,
    #                 style="filled",
    #                 fillcolor='lightgrey'))
    #     edge = graph.get_edge(('s0', 's1'))

    #     edge_label = edge[0].get_label()
    #     print(edge_label)
    #     return graph

    # def _get_edge_details(self, graph, edge_name):
    #     edge = graph.get_edge(edge_name)

    #     try:
    #         edge_label = edge[0].get_label()

    #     except KeyError or IndexError:
    #         edge_label = edge.get_label()
