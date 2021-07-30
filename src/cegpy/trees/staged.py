from ..utilities.util import Util
from ..trees.event import EventTree
from fractions import Fraction
from operator import add, sub
from IPython.display import Image
from IPython import get_ipython
import random
import scipy.special
import math
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
        self._mean_posterior_probs = []
        self._merged_situations = []
        self._stage_colours = []
        self._sort_count = 0
        self._colours_for_situations = []
        logger.debug("Starting Staged Tree")

        # Call event tree init to generate event tree
        super().__init__(params)

        # # Make params available to all local functions
        # self.params = dict(params)
        # logger.debug("Params passed to class are:")
        # logger.debug(self.params.items())
        # self._store_params()

    def _store_params(self, prior, alpha, hyperstage) -> None:
        """User has passed in AHC params, this function processes them,
        and generates any default AHC params if required."""
        if prior:
            if alpha:
                self.alpha = None
                logging.warning("Params Warning!! When prior is given, \
                                alpha is not required!")
        else:
            if alpha is None:
                self.alpha = self._calculate_default_alpha()
                logging.warning("Params Warning!! Neither prior nor alpha\
                                were provided. Using default alpha value of\
                                %d.", self.alpha)
            else:
                self.alpha = alpha

            # No matter what alpha is, generate default prior
            self.prior = self._create_default_prior(self.alpha)

        if hyperstage is None:
            self.hyperstage = self._create_default_hyperstage()
        else:
            self.hyperstage = hyperstage

    def _calculate_default_alpha(self) -> int:
        """If no alpha is given, a default value is calculated.
        The value is calculated by determining the maximum number
        of categories that any one variable has"""
        logger.info("Calculating default prior")
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

        logger.info("Generating default prior")
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
        logger.info("Creating default hyperstage")
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
        logger.info("Creating edge countset")
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

    def _calculate_posterior(self, prior) -> list:
        '''calculating the posterior edge counts for the AHC method.
        The posterior for each edge is obtained as the sum of its prior
        and edge count. Here we do this such that the posterior is a
        list of lists. Each list gives the posterior along the edges
        emanating from a specific vertex. The indexing is the same as
        self.edge_countset and self.situations'''
        logger.info("Calculating posterior")
        posterior = []
        edge_countset = self.get_edge_countset()
        for index in range(0, len(prior)):
            posterior.append(
                list(map(add, prior[index], edge_countset[index]))
            )
        return posterior

    def _calculate_lg_of_sum(self, array) -> float:
        '''function to calculate log gamma of the sum of an array'''
        array = [float(x) for x in array]
        return scipy.special.gammaln(sum(array))

    def _calculate_sum_of_lg(self, array) -> float:
        '''function to calculate log gamma of each element of an array'''
        return sum([scipy.special.gammaln(float(x)) for x in array])

    def _calculate_initial_loglikelihood(self, prior, posterior) -> float:
        '''calculating log likelihood given a prior and posterior'''
        # Calculate prior contribution
        logger.info("Calculating initial loglikelihood")
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

    def _check_issubset(self, item, hyperstage) -> bool:
        '''function to check if two situations belong to the same set in the
        hyperstage'''
        return any(set(item).issubset(element) for element in hyperstage)

    def _calculate_bayes_factor(self, prior1, posterior1,
                                prior2, posterior2) -> float:
        '''calculates the bayes factor comparing two models which differ in
        only one stage'''
        new_prior = list(map(add, prior1, prior2))
        new_posterior = list(map(add, posterior1, posterior2))
        return (
            self._calculate_lg_of_sum(new_prior)
            - self._calculate_lg_of_sum(prior1)
            - self._calculate_lg_of_sum(prior2)
            - self._calculate_lg_of_sum(new_posterior)
            + self._calculate_lg_of_sum(posterior1)
            + self._calculate_lg_of_sum(posterior2)
            + self._calculate_sum_of_lg(new_posterior)
            - self._calculate_sum_of_lg(posterior1)
            - self._calculate_sum_of_lg(posterior2)
            - self._calculate_sum_of_lg(new_prior)
            + self._calculate_sum_of_lg(prior1)
            + self._calculate_sum_of_lg(prior2)
        )

    def _sort_list(self, a_list_of_lists) -> list:
        '''function to sort a list of lists to remove repetitions'''

        for l1_idx in range(0, len(a_list_of_lists)):
            for l2_idx in range(l1_idx+1, len(a_list_of_lists)):
                list_1 = a_list_of_lists[l1_idx]
                list_2 = a_list_of_lists[l2_idx]
                lists_intersect = set(list_1) & set(list_2)

                if lists_intersect:
                    lists_union = list(set(list_1) | set(list_2))
                    # new_list_of_lists.append(lists_union)
                    a_list_of_lists[l1_idx] = []
                    a_list_of_lists[l2_idx] = lists_union

        new_list_of_lists = [
            elem for elem in a_list_of_lists
            if elem != []
        ]

        if new_list_of_lists == a_list_of_lists:
            return new_list_of_lists
        else:
            return self._sort_list(new_list_of_lists)

    def _calculate_mean_posterior_probs(self, probs) -> list:
        '''Iterates through array of lists, calculates mean
        posterior probabilities'''
        mean_posterior_probs = []
        for arr in probs:
            total = sum(arr)
            mean_posterior_probs.append(
                [round(element/total, 3) for element in arr]
            )
        return mean_posterior_probs

    def _execute_AHC_algoritm(self):
        prior = self.get_prior()
        hyperstage = self.get_hyperstage()
        posterior = self.get_posterior()
        loglikelihood = self._calculate_initial_loglikelihood(prior, posterior)
        posterior_probs = self.get_posterior()
        situ = self.get_situations()
        merged_situation_list = []
        bayesfactor_score = 1
        logger.info(" ----- Starting main loop of AHC algorithm -----")
        logger.info("Prior is length %d" % len(prior))
        # print("Progress:   0%")
        while bayesfactor_score > 0:
            local_merges = []
            local_scores = []

            # perc = int(math.floor(len(prior) / 100))
            index = 0

            for sit_1 in range(len(prior)):
                # if(index % perc) == 0:
                #     current_perc = index / perc
                #     print("%d" % current_perc)

                if all(items == 0 for items in posterior[sit_1]) is False:
                    model1 = [prior[sit_1], posterior[sit_1]]
                    for sit_2 in range(sit_1+1, len(prior)):
                        is_subset = self._check_issubset(
                            [situ[sit_1], situ[sit_2]],
                            hyperstage
                        )

                        any_non_zero = all(
                            items == 0 for items in posterior[sit_2]
                        )

                        if is_subset and not any_non_zero:
                            model2 = [prior[sit_2], posterior[sit_2]]
                            local_scores.append(
                                self._calculate_bayes_factor(
                                    *model1, *model2
                                )
                            )

                            local_merges.append([sit_1, sit_2])

                index += 1

            if local_scores != [] and max(local_scores) > 0:
                bayesfactor_score = max(local_scores)

                merged_situation_list.append(
                    local_merges[local_scores.index(bayesfactor_score)]
                )

                change_idx = merged_situation_list[-1]

                prior[change_idx[0]] = list(
                    map(
                        add,
                        prior[change_idx[0]],
                        prior[change_idx[1]]
                    )
                )
                posterior[change_idx[0]] = list(
                    map(
                        add,
                        posterior[change_idx[0]],
                        posterior[change_idx[1]]
                    )
                )

                prior[change_idx[1]] = [0] * len(prior[change_idx[0]])
                posterior[change_idx[1]] = [0] * len(prior[change_idx[0]])

                posterior_probs[change_idx[0]] = posterior[change_idx[0]]
                posterior_probs[change_idx[1]] = posterior[change_idx[0]]

                loglikelihood += bayesfactor_score

            elif max(local_scores) <= 0:
                bayesfactor_score = 0

            logger.debug('--Current AHC score: %d' % bayesfactor_score)

        return posterior_probs, loglikelihood, merged_situation_list

    def _create_merged_situations(self, merged_situation_indexes):
        """AHC algorithm creates a list of indexes to the situations list.
        This function takes those indexes and creates a new list which is
        in a string representation of nodes."""
        self._sort_count = 0
        list_of_merged_situations = self._sort_list(merged_situation_indexes)
        merged_situations = []
        situ = self.get_situations()
        for stage in list_of_merged_situations:
            merged_situations.append(
                [situ[index] for index in stage]
            )
        return merged_situations

    def _generate_stage_colours(self, number):
        '''generating unique colours for the staged tree and event tree.
        This function is seeded so that colours are the same on multiple
        runs of the code'''
        random.seed("12345")
        _HEX = '0123456789abcdef'

        def startcolor():
            return '#' + ''.join(random.choice(_HEX) for _ in range(6))
        colours = []
        for index in range(0, number):
            newcolour = startcolor()
            while newcolour in colours:
                newcolour = startcolor()
            colours.append(newcolour)
        return colours

    def _generate_colours_for_situations(self):
        """Colours each stage of the tree with an individual colour"""
        number_of_stages = len(self._merged_situations)
        stage_colours = Util.generate_colours(number_of_stages)
        self._stage_colours = stage_colours
        colours_for_situations = {}

        for node in self.nodes:
            stage_logic_values = [
                (node in stage) for stage in self._merged_situations
            ]

            if all(value == (False) for value in stage_logic_values):
                colours_for_situations[node] = 'lightgrey'
            else:
                colour_index = stage_logic_values.index((True))
                colours_for_situations[node] = \
                    stage_colours[colour_index]

        return colours_for_situations

    def calculate_AHC_transitions(self, prior=None,
                                  alpha=None, hyperstage=None):
        '''Bayesian Agglommerative Hierarchical Clustering algorithm
        implementation. It returns a list of lists of the situations which
        have been merged together, the likelihood of the final model and
        the mean posterior conditional probabilities of the stages.'''
        logger.info("\n\n --- Starting AHC Algorithm ---")

        self._store_params(prior, alpha, hyperstage)

        posterior_probs, loglikelihood, \
            merged_situation_indexes = self._execute_AHC_algoritm()

        self._mean_posterior_probs = \
            self._calculate_mean_posterior_probs(posterior_probs)

        self._merged_situations = \
            self._create_merged_situations(merged_situation_indexes)

        self._colours_for_situations = \
            self._generate_colours_for_situations()

        return {
            "Merged Situations": self._merged_situations,
            "Loglikelihood": loglikelihood,
            "Mean Posterior Probabilities": self._mean_posterior_probs
        }

    def create_figure(self, filename):
        """Draws the event tree for the process described by the dataset,
        and saves it to <filename>.png"""

        if self._colours_for_situations:
            filename, filetype = Util.generate_filename_and_mkdir(filename)
            logger.info("--- generating graph ---")
            graph = self._generate_graph(colours=self._colours_for_situations)
            logger.info("--- writing " + filetype + " file ---")
            graph.write(str(filename), format=filetype)

            if get_ipython() is None:
                return None
            else:
                logger.info("--- Exporting graph to notebook ---")
                return Image(graph.create_png())

        else:
            logger.error("----- PLEASE RUN AHC ALGORITHM before trying to \
                export graph -----")
            return None

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
