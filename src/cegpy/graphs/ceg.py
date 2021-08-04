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


            # child_node = {
            #     'name': edge_key[1][1],
            #     'label': edge_key[0][-1],
            #     'count': event_tree[edge_key] + float(prior[idx])
            # }

            # # Add children to node
            # try:
            #     # Modify existing node
            #     # add child to the node
            #     graph['nodes'][node_name]['children'].append(child_node)
            # except KeyError:
            #     # Create the new node
            #     graph['nodes'][node_name] = self._create_new_node()
            #     # add child to the node
            #     graph['nodes'][node_name]['children'].append(child_node)

            # # Create Child node, add parent information
            # child_name = child_node['name']
            # try:
            #     # If node already exists, add parent node
            #     graph[child_node[child_name]]['parents'].append(node_name)
            # except KeyError:
            #     # Create the new node
            #     graph[child_name] = self._create_new_node()
            #     graph[child_name]['parents'].append(node_name)

        return graph

    def _flatten_list_of_lists(self, list_of_lists) -> list:
        flat_list = []
        for sublist in list_of_lists:
            flat_list = flat_list + sublist
        return flat_list

    def _create_new_node(self, nodes_to_merge=[], colour='lightgrey') -> dict:
        node = {
            'nodes_to_merge': nodes_to_merge,
            'colour': colour
        }
        return node

    def _create_new_edge(self, src='', dest='', label='', value=0.0) -> list:
        edge = {
            'src': src,
            'dest': dest,
            'label': label,
            'value': value
        }
        return edge

    def _graph(self):
        """
        Generates the parameters used to output the CEG figure.
        """
        self._create_graph_edges()
        self._create_graph_vertices()
        self._colour_graph_vertices()
        pass

    def _create_graph_edges(self):
        """
        Determines the edges of the CEG, along with their labels.
        """
        pass

    def _create_graph_vertices(self):
        """
        Identifies the positions, i.e. the vertices of the CEG.
        """
        pass

    def _colour_graph_vertices(self):
        """
        Assigns colours to each of the combined vertices.
        """
        pass

    def create_figure(self):
        pass

    def _create_figure_graph(self):
        pass

    def _create_figure_edges(self):

        pass

    def _create_figure_vertices(self):

        pass

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
