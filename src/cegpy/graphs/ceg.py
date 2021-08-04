# from ..trees.staged import StagedTree
# import pydotplus as pdp
# from operator import add


class ChainEventGraph(object):
    """
    Class: Chain Event Graph


    Input: Staged tree object (StagedTree)
    Output: Chain event graphs
    """
    def __init__(self, staged_tree=None) -> None:
        # self.st = StagedTree()
        self.st = staged_tree
        if self.st.get_AHC_output() == {}:
            raise ValueError("Run staged tree AHC transitions first.")
        self.ceg = self._create_graph_representation()

        pass

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

            # Add src node to graph dict:
            try:
                graph['nodes'][edge['src']]
            except KeyError:
                graph['nodes'][edge['src']] = self._create_new_node()
            # Add dest node
            try:
                graph['nodes'][edge['dest']]
            except KeyError:
                graph['nodes'][edge['dest']] = self._create_new_node()

            # Add edge to graph dict:
            new_edge_key = edge_key[1]
            try:
                graph['edges'][new_edge_key].append(edge)
            except KeyError:
                graph['edges'][new_edge_key] = []
                graph['edges'][new_edge_key].append(edge)

        return graph

    def _flatten_list_of_lists(self, list_of_lists) -> list:
        flat_list = []
        for sublist in list_of_lists:
            flat_list = flat_list + sublist
        return flat_list

    def _create_new_node(self, root=False, sink=False,
                         nodes_to_merge=[], colour='lightgrey') -> dict:
        """
        Generates default format of Node dictionary
        """
        node = {
            'root': root,
            'sink': sink,
            'nodes_to_merge': nodes_to_merge,
            'colour': colour
        }
        return node

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

    def _trim_leaves_from_graph(self, graph) -> dict:
        leaves = self.st.get_leaves()
        graph['nodes']['w_inf'] = self._create_new_node(sink=True)
        sink_node = 'w_inf'
        # Check which nodes have been identified as leaves
        for leaf in leaves:
            edges_to_delete = []
            edges_to_add = {}
            # In edge list, look for edges that terminate on this leaf
            for edge_list_key in graph['edges'].keys():
                if edge_list_key[1] == leaf:
                    new_edge_list = []
                    # Each edge key may have multiple edges associate with it
                    for edge in graph['edges'][edge_list_key]:
                        new_edge = edge
                        new_edge['dest'] = sink_node
                        new_edge_list.append(new_edge)

                    # add modified edge to the dictionary
                    edges_to_add[(edge_list_key[0], sink_node)] = \
                        new_edge_list

                    # remove out of date edges from the dictionary
                    edges_to_delete.append(edge_list_key)

            # remove leaf node from the graph
            del graph['nodes'][leaf]
            # clean up old edges
            for edge in edges_to_delete:
                del graph['edges'][edge]

            graph['edges'] = {**graph['edges'], **edges_to_add}

        return graph

    def _merge_nodes(self) -> dict:
        return self.ceg

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
