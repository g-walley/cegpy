from collections import defaultdict
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
logger = logging.getLogger('cegpy.event_tree')


class EventTree(nx.MultiDiGraph):
    """
    Class for event trees.

    This class extends the networkx DiGraph class to allow the creation
    of event trees from data provided in a pandas dataframe.

    A DiGraph stores nodes and edges with optional data, or attributes.

    DiGraphs hold directed edges.  Self loops are allowed but multiple
    (parallel) edges are not.

    Nodes can be arbitrary (hashable) Python objects with optional
    key/value attributes. By convention `None` is not used as a node.

    Edges are represented as links between nodes with optional
    key/value attributes.

    Parameters
    ----------
    dataframe : Pandas dataframe (required)
        Dataframe containing variables as column headers, with event
        name strings in each cell. These event names will be used to
        create the edges of the event tree. Counts of each event will
        be extracted and attached to each edge.

    sampling_zero_paths: list of tuples containing paths to sampling
        zeros.
        Format is as follows: \
            [('edge_1',), ('edge_1', 'edge_2'), ...]

    incoming_graph_data : input graph (optional, default: None)
        Data to initialize graph.  If None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.  If the corresponding optional Python
        packages are installed the data can also be a NumPy matrix
        or 2d ndarray, a SciPy sparse matrix, or a PyGraphviz graph.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    StagedTrees \n
    ChainEventGraph

    Examples
    --------
    >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    >>> G = nx.Graph(name="my graph")
    >>> e = [(1, 2), (2, 3), (3, 4)]  # list of edges
    >>> G = nx.Graph(e)

    Arbitrary graph attribute pairs (key=value) may be assigned

    >>> G = nx.Graph(e, day="Friday")
    >>> G.graph
    {'day': 'Friday'}

    """
    def __init__(self, dataframe, sampling_zero_paths=None,
                 incoming_graph_data=None, var_order=None, **attr) -> None:
        """Initialize an event tree graph with edges, name, or graph attributes.
        This class extends the networkx DiGraph class to allow the creation
        of event trees from data provided in a pandas dataframe.

        Parameters
        ----------
        dataframe : Pandas dataframe (required)
            Dataframe containing variables as column headers, with event
            name strings in each cell. These event names will be used to
            create the edges of the event tree. Counts of each event will
            be extracted and attached to each edge.

        sampling_zero_paths: list of tuples containing paths to sampling
            zeros.
            Format is as follows: \
                [('edge_1',), ('edge_1', 'edge_2'), ...]

        incoming_graph_data : input graph (optional, default: None)
            Data to initialize graph.  If None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object.  If the corresponding optional Python
            packages are installed the data can also be a NumPy matrix
            or 2d ndarray, a SciPy sparse matrix, or a PyGraphviz graph.
        
        var_order : ordered list of variable names. (optional, default order 
            of variables in the event tree adopted from the order of columns in 
            the dataframe). 

        attr : keyword arguments, optional (default= no attributes)
            Attributes to add to graph as key=value pairs.

        See Also
        --------
        convert

        Examples
        --------
        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G = nx.Graph(name="my graph")
        >>> e = [(1, 2), (2, 3), (3, 4)]  # list of edges
        >>> G = nx.Graph(e)

        Arbitrary graph attribute pairs (key=value) may be assigned

        >>> G = nx.Graph(e, day="Friday")
        >>> G.graph
        {'day': 'Friday'}

        """
        logger.info('Initialising')
        # Initialise Networkx DiGraph class
        super().__init__(incoming_graph_data, **attr)
        self._sampling_zero_paths = None
        self.sampling_zeros = sampling_zero_paths

        # Paths sorted alphabetically in order of length
        self._sorted_paths = defaultdict(int)

        # pandas dataframe passed via parameters
        if var_order is not None:
            self.dataframe = dataframe[var_order]
        else:
            self.dataframe = dataframe
        self.__construct_event_tree()
        logger.info('Initialisation complete!')

    @property
    def root(self) -> str:
        """Root node of the event tree.
        Currently hard coded to 's0'"""
        return 's0'

    @property
    def variables(self) -> list:
        """The column headers of the dataset"""
        vars = list(self._dataframe.columns)
        logger.info('Variables extracted from dataframe were:')
        logger.info(vars)
        return vars

    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, value: pd.DataFrame):
        if isinstance(value, pd.DataFrame):
            self._dataframe = value
        else:
            raise ValueError(
                "Package currently only supports Pandas DataFrame"
                " objects provided as the dataframe")

    @property
    def sampling_zeros(self):
        if self._sampling_zero_paths is None:
            logger.info("EventTree.sampling_zero_paths \
                    has not been set.")
        return self._sampling_zero_paths

    @sampling_zeros.setter
    def sampling_zeros(self, sz_paths):
        """Use this function to set the sampling zero paths.
        If different to previous value, will re-generate the event tree."""
        if sz_paths is None:
            self._sampling_zero_paths = None
        else:
            # checkes if the user has inputted sz paths correctly
            sz_paths = self.__check_sampling_zero_paths_param(sz_paths)

            if sz_paths:
                self._sampling_zero_paths = sz_paths
            else:
                error_str = "Parameter 'sampling_zero_paths' not in expected format. \
                             Should be a list of tuples like so:\n \
                             [('edge_1',), ('edge_1', 'edge_2'), ...]"
                raise ValueError(error_str)

    @property
    def situations(self) -> list:
        """List of situations of the tree.
        (non-leaf nodes)"""
        return [
            node for node, out_degree in self.out_degree
            if out_degree != 0
        ]

    @property
    def leaves(self) -> list:
        """List of leaves of the tree."""
        # if not already generated, create self.leaves
        return [
            node for node, out_degree in self.out_degree
            if out_degree == 0
        ]

    @property
    def edge_counts(self) -> dict:
        '''list of counts along edges. Indexed same as edges and edge_labels'''
        return nx.get_edge_attributes(self, 'count')

    @property
    def categories_per_variable(self) -> dict:
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
        self._catagories_per_variable = {}
        nans_filtered = False

        for var in self.variables:
            categories = set(self._dataframe[var].unique().tolist())
            # remove nan with pandas
            pd_filtered_categories = {x for x in categories if pd.notna(x)}
            if pd_filtered_categories != categories:
                nans_filtered = True

            # remove any string nans that might have made it in.
            filtered_cats = pd_filtered_categories - categories_to_ignore
            if pd_filtered_categories != filtered_cats:
                nans_filtered = True

            self._catagories_per_variable[var] = len(filtered_cats)

        if nans_filtered:
            display_nan_warning()

        return self._catagories_per_variable

    @property
    def dot_graph(self):
        return self._generate_dot_graph()

    def _generate_dot_graph(self):
        node_list = list(self)
        graph = pdp.Dot(graph_type='digraph', rankdir='LR')
        for edge, count in self.edge_counts.items():
            edge_details = str(edge[2]) + '\n' + str(count)

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
            try:
                fill_colour = self.nodes[node]['colour']
            except KeyError:
                fill_colour = 'lightgrey'
            label = "<" + node[0] + "<SUB>" + node[1:] + "</SUB>" + ">"
            graph.add_node(
                pdp.Node(
                    name=node,
                    label=label,
                    style="filled",
                    fillcolor=fill_colour))
        return graph

    def create_figure(self, filename):
        """Draws the event tree for the process described by the dataset,
        and saves it to "<filename>.filetype". Supports any filetype that
        graphviz supports. e.g: "event_tree.png" or "event_tree.svg" etc.
        """
        filename, filetype = Util.generate_filename_and_mkdir(filename)
        logger.info("--- generating graph ---")
        graph = self.dot_graph
        logger.info("--- writing " + filetype + " file ---")
        graph.write(str(filename), format=filetype)

        if get_ipython() is None:
            return None
        else:
            logger.info("--- Exporting graph to notebook ---")
            return Image(graph.create_png())

    def __create_unsorted_paths_dict(self) -> defaultdict:
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

    def __create_path_dict_entries(self):
        '''Create path dict entries for each path, including the
        sampling zero paths if any.
        Each path is an ordered sequence of edge labels starting
        from the root.
        The keys in the dict are ordered alphabetically.
        Also calls the method self._sampling_zeros to ensure
        manually added path format is correct.
        Added functionality to remove NaN/null edge labels
        assuming they are structural zeroes'''
        unsorted_paths = self.__create_unsorted_paths_dict()

        if self.sampling_zeros is not None:
            unsorted_paths = Util.create_sampling_zeros(
                self.sampling_zeros, unsorted_paths)

        depth = len(max(list(unsorted_paths.keys()), key=len))
        keys_of_list = list(unsorted_paths.keys())
        sorted_keys = []
        for deep in range(0, depth + 1):
            unsorted_mini_list = [key for key in keys_of_list if
                                  len(key) == deep]
            sorted_keys = sorted_keys + sorted(unsorted_mini_list)

        for key in sorted_keys:
            self._sorted_paths[key] = unsorted_paths[key]

        node_list = self.__create_node_list_from_paths(self._sorted_paths)
        self.add_nodes_from(node_list)

    def __check_sampling_zero_paths_param(self, sampling_zero_paths) -> list:
        """Check param 'sampling_zero_paths' is in the correct format"""
        for tup in sampling_zero_paths:
            if not isinstance(tup, tuple):
                return None
            else:
                if not Util.check_tuple_contains_strings(tup):
                    return None

        return sampling_zero_paths

    def __create_node_list_from_paths(self, paths) -> list:
        """Creates list of all nodes: includes root, situations, leaves"""
        node_list = [self.root]

        for vertex_number, _ in enumerate(list(paths.keys()), start=1):
            node_list.append('s%d' % vertex_number)

        return node_list

    def __construct_event_tree(self):
        """Constructs event_tree DiGraph.
        Takes the paths, and adds all the nodes and edges to the Graph"""

        logger.info('Starting construction of event tree')
        self.__create_path_dict_entries()
        # Taking a list of a networkx graph object (self) provides a list
        # of all the nodes
        node_list = list(self)

        # Work through the sorted paths list to build the event tree.
        edge_labels_list = ['root']
        for path, count in list(self._sorted_paths.items()):
            path = list(path)
            edge_labels_list.append(path)
            if path[:-1] in edge_labels_list:
                path_edge_comes_from = edge_labels_list.index(path[:-1])
                self.add_edge(
                    u_for_edge=node_list[path_edge_comes_from],
                    v_for_edge=node_list[edge_labels_list.index(path)],
                    key=path[-1],
                    count=count
                )
            else:
                self.add_edge(
                    u_for_edge=node_list[0],
                    v_for_edge=node_list[edge_labels_list.index(path)],
                    key=path[-1],
                    count=count
                )
