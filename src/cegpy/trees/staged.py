from ..trees.event import EventTree
from fractions import Fraction
from operator import add, sub
import scipy.special
# from ..utilities.util import Util
import logging

logger = logging.getLogger('pyceg.staged_tree')


class StagedTree(EventTree):
    def __init__(self, params) -> None:
        self.prior = None  # List of lists
        self.alpha = None  # int
        self.hyperstage = None
        self.edge_countset = None
        self.posterior = None
        logger.debug("Starting Staged Tree")

        # Call event tree init to generate event tree
        super().__init__(params)

        # Make params available to all local functions
        self.params = dict(params)
        logger.debug("Params passed to class are:")
        logger.debug(self.params.items())
        self._store_params()

    def _store_params(self) -> None:
        self.prior = self.params.get("prior")
        self.alpha = self.params.get("alpha")
        self.hyperstage = self.params.get("hyperstage")

        if self.prior:
            if self.alpha:
                self.alpha = None
                logging.warning("Params Warning!! When prior is given, \
                                alpha is not required!")
        else:
            if self.alpha is None:
                self.alpha = self._calculate_default_alpha()
                logging.warning("Params Warning!! Neither prior nor alpha\
                                were provided. Using default alpha value of\
                                %d.", self.alpha)
            # No matter what alpha is, generate default prior
            self.prior = self._create_default_prior(self.alpha)

        if self.hyperstage is None:
            self.hyperstage = self._create_default_hyperstage()

    def _calculate_default_alpha(self) -> int:
        """If no alpha is given, a default value is calculated.
        The value is calculated by determining the maximum number
        of categories that any one variable has"""
        max_count = max(list(self.get_categories_per_variable().values()))
        return max_count

    def _create_default_prior(self, alpha) -> list:
        """default prior set for the AHC method using the mass conservation property.
        That is, the alpha param is the phantom sample starting at
        the root, and it is spread equally across all edges along the tree.
        (see chapter 5 of Collazo, Gorgen & Smith 'Chain Event Graphs', 2018)
        The prior is a list of lists. Each list gives the prior along the
        edges of a specific situation.
        Indexed same as self.situations & self.egde_countset"""
        default_prior = [0] * len(self.get_situations())
        sample_size_at_node = dict()

        # Root node is assigned phantom sample (alpha)
        if isinstance(alpha, float):
            alpha = Fraction.from_float(alpha)
        elif isinstance(alpha, int) or isinstance(alpha, str):
            alpha = Fraction.from_float(float(alpha))
        else:
            logger.warning("Prior generator param alpha is in a strange format.\
                            ..")

        sample_size_at_node[self.get_root()] = alpha

        for node_idx, node in enumerate(self.get_situations()):
            # How many nodes emanate from the current node?
            number_of_emanating_nodes = self.get_emanating_nodes().count(node)
            # Divide the sample size from the current node equally among
            # emanating nodes
            equal_distribution_of_sample = \
                sample_size_at_node[node] / number_of_emanating_nodes
            default_prior[node_idx] = \
                [equal_distribution_of_sample] * number_of_emanating_nodes

            relevant_terminating_nodes = [
                self.get_terminating_nodes()[self.get_edges().index(edge_pair)]
                for edge_pair in self.get_edges() if edge_pair[0] == node
            ]

            for terminating_node in relevant_terminating_nodes:
                sample_size_at_node[terminating_node] = \
                 equal_distribution_of_sample

        return default_prior

    def _create_default_hyperstage(self) -> list:
        '''Generates default hyperstage for the AHC method.
        A hyperstage is a list of lists such that two situaions can be in the
        same stage only if there are elements of the same list for some list
        in the hyperstage.
        The default is to allow all situations with the same number of
        outgoing edges and the same edge labels to be in a common list. '''
        hyperstage = []
        info_of_edges = []
        edges = self.get_edges()
        edge_labels = self.get_edge_labels()
        situations = self.get_situations()
        emanating_nodes = self.get_emanating_nodes()

        for node in situations:
            edge_indices = [
                edges.index(edge) for edge in self.edges
                if edge[0] == node
            ]

            labels = [edge_labels[x][-1] for x in edge_indices]
            labels.sort()

            info_of_edges.append(
                [emanating_nodes.count(node), labels]
            )

        sorted_info = []
        for info in info_of_edges:
            if info not in sorted_info:
                sorted_info.append(info)

        for info in sorted_info:
            situations_with_value_edges = []
            for idx, situation in enumerate(situations):
                if info_of_edges[idx] == info:
                    situations_with_value_edges.append(situation)
            hyperstage = hyperstage + [situations_with_value_edges]

        return hyperstage

    def _create_edge_countset(self) -> list:
        '''Each element of list contains a list with counts along edges emanating from
        a specific situation. Indexed same as self.situations'''
        edge_countset = []
        situations = self.get_situations()
        edges = self.get_edges()
        edge_counts = self.get_edge_counts()
        term_nodes = self.get_terminating_nodes()

        for node in situations:
            edgeset = [
                edge_pair[1] for edge_pair in edges
                if edge_pair[0] == node
            ]

            edge_countset.append([
                edge_counts[term_nodes.index(vertex)]
                for vertex in edgeset
            ])
        return edge_countset

    def _calculate_posterior(self, prior):
        '''calculating the posterior edge counts for the AHC method.
        The posterior for each edge is obtained as the sum of its prior
        and edge count. Here we do this such that the posterior is a
        list of lists. Each list gives the posterior along the edges
        emanating from a specific vertex. The indexing is the same as
        self.edge_countset and self.situations'''
        posterior = []
        edge_countset = self.get_edge_countset()
        for index in range(0, len(prior)):
            posterior.append(
                list(map(add, prior[index], edge_countset[index]))
            )
        return posterior

    def _calculate_lg_of_sum(self, array):
        '''function to calculate log gamma of the sum of an array'''
        array = [float(x) for x in array]
        return scipy.special.gammaln(sum(array))

    def _calculate_sum_of_lg(self, array):
        '''function to calculate log gamma of each element of an array'''
        return sum([scipy.special.gammaln(float(x)) for x in array])

    def _calculate_loglikehood(self, prior, posterior):
        '''calculating log likelihood given a prior and posterior'''
        # Calculate prior contribution

        pri_lg_of_sum = [
            self._calculate_lg_of_sum(elem) for elem in prior
        ]
        pri_sum_of_lg = [
            self._calculate_sum_of_lg(elem) for elem in prior
        ]
        pri_contribution = list(map(sub, pri_lg_of_sum, pri_sum_of_lg))

        # Calculate posterior contribution
        post_lg_of_sum = [
            self._calculate_lg_of_sum(elem) for elem in posterior
        ]
        post_sum_of_lg = [
            self._calculate_sum_of_lg(elem) for elem in posterior
        ]
        post_contribution = list(map(sub, post_sum_of_lg, post_lg_of_sum))

        return (sum(pri_contribution) + sum(post_contribution))

    def get_prior(self):  # TODO: INCLUDE -> when type is known
        return self.prior

    def get_alpha(self):
        return self.alpha

    def get_hyperstage(self):
        return self.hyperstage

    def get_edge_countset(self):
        if not self.edge_countset:
            self.countset = self._create_edge_countset()

        return self.countset

    def get_posterior(self):
        if not self.posterior:
            prior = self.get_prior()
            self.posterior = self._calculate_posterior(prior)

        return self.posterior
