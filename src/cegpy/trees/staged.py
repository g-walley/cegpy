from ..trees.event import EventTree
from fractions import Fraction
# from ..utilities.util import Util
import logging

logger = logging.getLogger('pyceg.staged_tree')


class StagedTree(EventTree):
    def __init__(self, params) -> None:
        self.prior = None  # List of lists
        self.alpha = None  # int
        logger.debug("Starting Staged Tree")

        # Call event tree init to generate event tree
        super().__init__(params)

        # Make params available to all local functions
        self.params = dict(params)
        logger.debug("Params passed to class are:")
        logger.debug(self.params.items())
        self._check_and_store_params()

    def _check_and_store_params(self) -> None:
        self.prior = self._check_param_prior(self.params.get("prior"))
        self.alpha = self._check_param_alpha(self.params.get("alpha"))

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
            self.prior = self._generate_default_prior(self.alpha)

    def _calculate_default_alpha(self) -> int:
        """If no alpha is given, a default value is calculated.
        The value is calculated by determining the maximum number
        of categories that any one variable has"""
        max_count = max(list(self.get_categories_per_variable().values()))
        return max_count

    def _generate_default_prior(self, alpha) -> list:
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

    def get_prior(self):  # TODO: INCLUDE -> when type is known
        return self.prior

    def get_alpha(self):
        return self.alpha

    def _check_param_prior(self, prior):
        # TODO: Implement Prior param check when you know what it should be
        return prior

    def _check_param_alpha(self, alpha):
        # TODO: Implement alpha param check when you know what it should be
        return alpha
