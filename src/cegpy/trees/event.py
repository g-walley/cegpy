from collections import defaultdict
# from typing import OrderedDict
# import pandas as pd
import numpy as np
import pydotplus as pdp
import logging
from ..utilities.util import Util
from IPython.display import Image
from IPython import get_ipython
import pandas as pd
import textwrap
import networkx as nx
# create logger object for this module
logger = logging.getLogger('pyceg.event_tree')


class EventTree(nx.DiGraph):
    """Creates event trees from pandas dataframe."""
    def __init__(self, dataframe, sampling_zero_paths=None,
                 incoming_graph_data=None, **attr) -> None:
        logger.info('Initialising')
        # Initialise Networkx DiGraph class
        super().__init__(incoming_graph_data, **attr)

        self._set_sampling_zero_paths(sampling_zero_paths)

        # Paths sorted alphabetically in order of length
        self.sorted_paths = defaultdict(int)

        # pandas dataframe passed via parameters
        self.dataframe = dataframe
        # Format of event_tree dict:

        self._construct_event_tree()
        logger.info('Initialisation complete!')

    @property
    def root(self) -> str:
        return 's0'

    @property
    def variables(self) -> list:
        try:
            return self._variables
        except AttributeError:
            self._variables = list(self.dataframe.columns)
            logger.info('Variables extracted from dataframe were:')
            logger.info(self._variables)
            return self._variables

    @property
    def sampling_zero_paths(self):
        if not self._sampling_zero_paths:
            logger.info("EventTree.sampling_zero_paths \
                        has not been set.")

        return self._sampling_zero_paths

    @property
    def edge_labels(self) -> dict:
        """Once event has been created, a dict of all
        edge labels can be obtained with this function"""
        return nx.get_edge_attributes(self, 'label')

    @property
    def situations(self) -> list:
        """Returns list of event tree situations.
        (non-leaf nodes)"""
        try:
            return self._situations
        except AttributeError:
            nodes = self.nodes
            self._situations = [
                node for node in nodes if node not in self.leaves
            ]

            return self._situations

    @property
    def leaves(self) -> list:
        """Returns leaves of the event tree."""
        # if not already generated, create self.leaves
        try:
            return self._leaves
        except AttributeError:
            edges = self.edges
            self._leaves = [
                edge_pair[1] for edge_pair in edges
                if edge_pair[1] not in self.emanating_nodes
            ]

            return self._leaves

    @property
    def emanating_nodes(self) -> list:
        """Returns list of situations where edges start."""
        # if not already generated, create self._emanating_nodes
        try:
            return self._emanating_nodes
        except AttributeError:
            edges = self.edges
            self._emanating_nodes = [edge_pair[0] for edge_pair in edges]
            return self._emanating_nodes

    @property
    def terminating_nodes(self) -> list:
        """Returns list of suations where edges terminate."""
        try:
            return self._terminating_nodes
        except AttributeError:
            self._terminating_nodes = [
                edge_pair[1] for edge_pair in self.edges
            ]

            return self._terminating_nodes

    @property
    def edge_counts(self) -> dict:
        '''list of counts along edges. Indexed same as edges and edge_labels'''
        return nx.get_edge_attributes(self, 'count')

    def _generate_graph(self, colours=None):
        node_list = list(self)
        graph = pdp.Dot(graph_type='digraph', rankdir='LR')
        for edge, label in self.edge_labels.items():
            count = self[edge[0]][edge[1]]['count']
            edge_details = label + '\n' + str(count)

            graph.add_edge(
                pdp.Edge(
                    edge[0],
                    edge[1],
                    label=edge_details,
                    labelfontcolor="#009933",
                    fontsize="10.0",
                    color="black"
                )
            )

        for node in node_list:
            if colours:
                fill_colour = colours[node]
            else:
                fill_colour = 'lightgrey'

            graph.add_node(
                pdp.Node(
                    name=node,
                    label=node,
                    style="filled",
                    fillcolor=fill_colour))
        return graph

    def create_figure(self, filename):
        """Draws the event tree for the process described by the dataset,
        and saves it to <filename>.png"""
        filename, filetype = Util.generate_filename_and_mkdir(filename)
        logger.info("--- generating graph ---")
        graph = self._generate_graph()
        logger.info("--- writing " + filetype + " file ---")
        graph.write(str(filename), format=filetype)

        if get_ipython() is None:
            return None
        else:
            logger.info("--- Exporting graph to notebook ---")
            return Image(graph.create_png())

    def get_categories_per_variable(self) -> dict:
        '''list of number of unique categories/levels for each variable
        (a column in the df)'''
        def display_nan_warning():
            logger.warning(
                textwrap.dedent(
                    """   --- NaNs found in the dataframe!! ---
                    cegpy assumes that NaNs are either structural zeros or
                    structural missing values.
                    Any non-structural missing values must be dealt with
                    prior to providing the dataset to any of the cegpy
                    functions. Any non-structural zeros should be explicitly
                    added into the cegpy objects.
                    --- See documentation for more information. ---"""
                )
            )

        categories_to_ignore = {"N/A", "NA", "n/a", "na", "NAN", "nan"}
        catagories_per_variable = {}
        nans_filtered = False

        for var in self.variables:
            categories = set(self.dataframe[var].unique().tolist())
            # remove nan with pandas
            pd_filtered_categories = {x for x in categories if pd.notna(x)}
            if pd_filtered_categories != categories:
                nans_filtered = True

            # remove any string nans that might have made it in.
            filtered_cats = pd_filtered_categories - categories_to_ignore
            if pd_filtered_categories != filtered_cats:
                nans_filtered = True

            catagories_per_variable[var] = len(filtered_cats)

        if nans_filtered:
            display_nan_warning()

        return catagories_per_variable

    def _set_sampling_zero_paths(self, sz_paths):
        """Use this function to set the sampling zero paths.
        If different to previous value, will re-generate the event tree."""
        if sz_paths is None:
            self._sampling_zero_paths = None
        else:
            # checkes if the user has inputted sz paths correctly
            sz_paths = self._check_sampling_zero_paths_param(sz_paths)

            if sz_paths:
                self._sampling_zero_paths = sz_paths
            else:
                error_str = "Parameter 'sampling_zero_paths' not in expected format. \
                             Should be a list of tuples like so:\n \
                             [('edge_1',), ('edge_1', 'edge_2'), ...]"
                if logger.getEffectiveLevel() is logging.DEBUG:
                    logger.debug(error_str)
                    raise ValueError(error_str)

    def _create_unsorted_paths_dict(self) -> defaultdict:
        """Creates and populates a dictionary of all paths provided in the dataframe,
        in the order in which they are given."""
        unsorted_paths = defaultdict(int)

        for variable_number in range(0, len(self.variables)):
            dataframe_upto_variable = self.dataframe.loc[
                :, self.variables[0:variable_number+1]]

            for row in dataframe_upto_variable.itertuples():
                row = row[1:]
                new_row = [edge_label for edge_label in row if
                           edge_label != np.nan and
                           str(edge_label) != 'NaN' and
                           str(edge_label) != 'nan' and
                           edge_label != '']
                new_row = tuple(new_row)

                # checking if the last edge label in row was nan. That would
                # result in double counting nan must be identified as string
                if (row[-1] != np.nan and str(row[-1]) != 'NaN' and
                   str(row[-1]) != 'nan' and row[-1] != ''):
                    unsorted_paths[new_row] += 1

        return unsorted_paths

    def _create_path_dict_entries(self):
        '''Create path dict entries for each path, including the
        sampling zero paths if any.
        Each path is an ordered sequence of edge labels starting
        from the root.
        The keys in the dict are ordered alphabetically.
        Also calls the method self._sampling_zeros to ensure
        manually added path format is correct.
        Added functionality to remove NaN/null edge labels
        assuming they are structural zeroes'''
        unsorted_paths = self._create_unsorted_paths_dict()

        if self.sampling_zero_paths is not None:
            unsorted_paths = Util.create_sampling_zeros(
                self.sampling_zero_paths, unsorted_paths)

        depth = len(max(list(unsorted_paths.keys()), key=len))
        keys_of_list = list(unsorted_paths.keys())
        sorted_keys = []
        for deep in range(0, depth + 1):
            unsorted_mini_list = [key for key in keys_of_list if
                                  len(key) == deep]
            sorted_keys = sorted_keys + sorted(unsorted_mini_list)

        for key in sorted_keys:
            self.sorted_paths[key] = unsorted_paths[key]

        node_list = self._create_node_list_from_paths(self.sorted_paths)
        self.add_nodes_from(node_list)

    def _check_sampling_zero_paths_param(self, sampling_zero_paths) -> list:
        """Check param 'sampling_zero_paths' is in the correct format"""
        for tup in sampling_zero_paths:
            if not isinstance(tup, tuple):
                return None
            else:
                if not Util.check_tuple_contains_strings(tup):
                    return None

        return sampling_zero_paths

    def _create_node_list_from_paths(self, paths) -> list:
        """Creates list of all nodes: includes root, situations, leaves"""
        node_list = [self.root]

        for vertex_number, _ in enumerate(list(paths.keys()), start=1):
            node_list.append('s%d' % vertex_number)

        return node_list

    def _construct_event_tree(self):
        """Constructs event_tree DiGraph.
        Takes the paths, and adds all the nodes and edges to the Graph"""

        logger.info('Starting construction of event tree')
        self._create_path_dict_entries()
        # Taking a list of a networkx graph object (self) provides a list
        # of all the nodes
        node_list = list(self)

        # Work through the sorted paths list to build the event tree.
        edge_labels_list = ['root']
        for path, count in list(self.sorted_paths.items()):
            path = list(path)
            edge_labels_list.append(path)
            if path[:-1] in edge_labels_list:
                path_edge_comes_from = edge_labels_list.index(path[:-1])
                self.add_edge(
                    u_of_edge=node_list[path_edge_comes_from],
                    v_of_edge=node_list[edge_labels_list.index(path)],
                    label=path[-1],
                    count=count
                )
            else:
                self.add_edge(
                    u_of_edge=node_list[0],
                    v_of_edge=node_list[edge_labels_list.index(path)],
                    label=path[-1],
                    count=count
                )
