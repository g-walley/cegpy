from copy import deepcopy
from fractions import Fraction
from operator import add, sub
from IPython.display import Image
from IPython import get_ipython
from itertools import combinations, chain, cycle
from typing import List, Tuple
import networkx as nx
import scipy.special
import logging
from ..utilities.util import Util
from ..trees.event import EventTree

logger = logging.getLogger('cegpy.staged_tree')


class StagedTree(EventTree):
    def __init__(
            self,
            dataframe,
            prior=None,
            alpha=None,
            hyperstage=None,
            sampling_zero_paths=None,
            incoming_graph_data=None,
            **attr) -> None:

        # Call event tree init to generate event tree
        super().__init__(
            dataframe,
            sampling_zero_paths,
            incoming_graph_data,
            **attr
        )

        self.__store_params(prior, alpha, hyperstage)
        self._mean_posterior_probs = []
        self._merged_situations = []
        self._stage_colours = []
        self._sort_count = 0
        self._colours_for_situations = []
        logger.debug("Starting Staged Tree")

    @property
    def prior(self):
        return nx.get_edge_attributes(self, 'prior')

    @prior.setter
    def prior(self, prior):
        offset = 0
        for node_idx, node_priors in enumerate(prior):
            node_name = ('s%d' % (node_idx + offset))
            while self.succ[node_name] == {}:
                offset += 1
                node_name = ('s%d' % (node_idx + offset))
            pass
            for edge_prior_idx, succ_key in enumerate(
                    self.succ[node_name].keys()):
                label = list(self.succ[node_name][succ_key])[0]
                self.edges[(node_name, succ_key, label)]['prior'] = \
                    node_priors[edge_prior_idx]

    @property
    def prior_list(self):
        """Priors provided as a list of lists"""
        prior_list = []
        prev_node = list(self.prior)[0][0]
        succ_list = []

        for edge, prior in self.prior.items():
            node = edge[0]
            if node != prev_node:
                prior_list.append(succ_list)
                succ_list = []
            succ_list.append(prior)
            prev_node = node

        if succ_list != []:
            prior_list.append(succ_list)
        return prior_list

    @property
    def posterior(self):
        '''Posterior is calculated such that the edge count is added
        to the prior for each edge.'''
        try:
            posterior = nx.get_edge_attributes(self, 'posterior')
            if posterior == {}:
                raise AttributeError('Posterior not yet set.')
            else:
                return posterior
        except AttributeError:
            for edge in self.edges:
                edge_dict = self.edges[edge]
                posterior = edge_dict['prior'] + edge_dict['count']
                edge_dict['posterior'] = posterior
            return nx.get_edge_attributes(self, 'posterior')

    @property
    def posterior_list(self):
        posterior_list = []
        prev_node = list(self.posterior)[0][0]
        succ_list = []

        for edge, posterior in self.posterior.items():
            node = edge[0]
            if node != prev_node:
                posterior_list.append(succ_list)
                succ_list = []
            succ_list.append(posterior)
            prev_node = node

        if succ_list != []:
            posterior_list.append(succ_list)
        return posterior_list

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def hyperstage(self):
        return self._hyperstage

    @hyperstage.setter
    def hyperstage(self, value):
        self._hyperstage = value

    @property
    def edge_countset(self):
        return self.__create_edge_countset()

    @property
    def ahc_output(self):
        return self._ahc_output

    @ahc_output.setter
    def ahc_output(self, value):
        self._ahc_output = value

    def __store_params(self, prior, alpha, hyperstage) -> None:
        """User has passed in AHC params, this function processes them,
        and generates any default AHC params if required."""
        if prior:
            if alpha:
                self.alpha = None
                logging.warning("Params Warning!! When prior is given, " +
                                "alpha is not required!")
        else:
            if alpha is None:
                self.alpha = self.__calculate_default_alpha()
                logging.warning("Params Warning!! Neither prior nor alpha " +
                                "were provided. Using default alpha " +
                                "value of %d.", self.alpha)
            else:
                self.alpha = alpha

            # No matter what alpha is, generate default prior
            self.prior = self.__create_default_prior(self.alpha)

        if hyperstage is None:
            self.hyperstage = self.__create_default_hyperstage()
        else:
            self.hyperstage = hyperstage

    def __calculate_default_alpha(self) -> int:
        """If no alpha is given, a default value is calculated.
        The value is calculated by determining the maximum number
        of categories that any one variable has"""
        logger.info("Calculating default prior")
        max_count = max(list(self.categories_per_variable.values()))
        return max_count

    def __create_default_prior(self, alpha) -> list:
        """default prior set for the AHC method using the mass conservation property.
        That is, the alpha param is the phantom sample starting at
        the root, and it is spread equally across all edges along the tree.
        (see chapter 5 of Collazo, Gorgen & Smith 'Chain Event Graphs', 2018)
        The prior is a list of lists. Each list gives the prior along the
        edges of a specific situation.
        Indexed same as self.situations & self.egde_countset"""

        logger.info("Generating default prior")
        default_prior = [0] * len(self.situations)
        sample_size_at_node = dict()

        # Root node is assigned phantom sample (alpha)
        if isinstance(alpha, float):
            alpha = Fraction.from_float(alpha)
        elif isinstance(alpha, int) or isinstance(alpha, str):
            alpha = Fraction.from_float(float(alpha))
        else:
            logger.warning("Prior generator param alpha is in a strange format.\
                            ..")

        sample_size_at_node[self.root] = alpha

        for node_idx, node in enumerate(self.situations):
            # How many nodes emanate from the current node?
            number_of_emanating_nodes = self.out_degree[node]

            # Divide the sample size from the current node equally among
            # emanating nodes
            equal_distribution_of_sample = \
                sample_size_at_node[node] / number_of_emanating_nodes
            default_prior[node_idx] = \
                [equal_distribution_of_sample] * number_of_emanating_nodes

            relevant_terminating_nodes = [
                edge[1] for edge in list(self.edges) if edge[0] == node
            ]

            for terminating_node in relevant_terminating_nodes:
                sample_size_at_node[terminating_node] = \
                 equal_distribution_of_sample

        return default_prior

    def __create_default_hyperstage(self) -> list:
        '''Generates default hyperstage for the AHC method.
        A hyperstage is a list of lists such that two situaions can be in the
        same stage only if there are elements of the same list for some list
        in the hyperstage.
        The default is to allow all situations with the same number of
        outgoing edges and the same edge labels to be in a common list. '''
        logger.info("Creating default hyperstage")
        hyperstage = []
        info_of_edges = []

        for node in self.situations:
            labels = [
                edge[2] for edge in self.edges
                if edge[0] == node
            ]
            labels.sort()

            info_of_edges.append([self.out_degree[node], labels])

        sorted_info = []
        for info in info_of_edges:
            if info not in sorted_info:
                sorted_info.append(info)

        for info in sorted_info:
            situations_with_value_edges = []
            for idx, situation in enumerate(self.situations):
                if info_of_edges[idx] == info:
                    situations_with_value_edges.append(situation)
            hyperstage = hyperstage + [situations_with_value_edges]

        return hyperstage

    def __create_edge_countset(self) -> list:
        '''Each element of list contains a list with counts along edges emanating from
        a specific situation. Indexed same as self.situations'''
        logger.info("Creating edge countset")
        edge_countset = []

        for node in self.situations:
            edge_countset.append([
                count for edge, count in self.edge_counts.items()
                if edge[0] == node
            ])
        return edge_countset

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

        sum_pri_contribution = sum(pri_contribution)
        sum_post_contribution = sum(post_contribution)
        return (sum_pri_contribution + sum_post_contribution)

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

    def _sort_list(self, list_of_tuples) -> list:
        '''function to sort a list of lists to remove repetitions'''

        for l1_idx in range(0, len(list_of_tuples)):
            for l2_idx in range(l1_idx+1, len(list_of_tuples)):
                tup_1 = list_of_tuples[l1_idx]
                tup_2 = list_of_tuples[l2_idx]
                tups_intersect = set(tup_1) & set(tup_2)

                if tups_intersect:
                    union = tuple(set(tup_1) | set(tup_2))
                    list_of_tuples[l1_idx] = []
                    list_of_tuples[l2_idx] = union

        new_list_of_tuples = [
            elem for elem in list_of_tuples
            if elem != []
        ]

        if new_list_of_tuples == list_of_tuples:
            return new_list_of_tuples
        else:
            return self._sort_list(new_list_of_tuples)

    def _calculate_mean_posterior_probs(
        self,
        merged_situations: List,
        posteriors: List,
    ) -> List:
        '''Iterates through array of lists, calculates mean
        posterior probabilities'''
        mean_posterior_probs = []

        for sit in self.situations:
            if sit not in list(chain(*merged_situations)):
                merged_situations.append((sit,))
        for stage in merged_situations:
            for sit in stage:
                sit_idx = self.situations.index(sit)
                if all(posteriors[sit_idx]) != 0:
                    stage_probs = posteriors[sit_idx]
                    break
                else:
                    stage_probs = []

            total = sum(stage_probs)
            mean_posterior_probs.append(
                [round(elem/total, 3) for elem in stage_probs]
            )

        return mean_posterior_probs

    def _independent_hyperstage_generator(
            self, hyperstage: List[List]) -> List[List[List]]:
        """Spit out the next hyperstage that can be dealt with
        independently."""
        new_hyperstages = [[hyperstage[0]]]

        for sublist in hyperstage[1:]:
            hs_to_add = [sublist]

            for hs in new_hyperstages.copy():
                for other_sublist in hs:
                    if not set(other_sublist).isdisjoint(set(sublist)):
                        hs_to_add.extend(hs)
                        new_hyperstages.remove(hs)

            new_hyperstages.append(hs_to_add)

        return new_hyperstages

    def _execute_AHC(self, hyperstage=None) -> Tuple[List, float, List]:
        """finds all subsets and scores them"""
        if hyperstage is None:
            hyperstage = deepcopy(self.hyperstage)

        priors = deepcopy(self.prior_list)
        posteriors = deepcopy(self.posterior_list)

        loglikelihood = self._calculate_initial_loglikelihood(
            priors, posteriors
        )

        merged_situation_list = []

        while True:
            hyperstage_combinations = [
                item for sub_hyper in hyperstage
                for item in combinations(sub_hyper, 2)
            ]

            newscores_list = [self._calculate_bayes_factor(
                priors[self.situations.index(sub_hyper[0])],
                posteriors[self.situations.index(sub_hyper[0])],
                priors[self.situations.index(sub_hyper[1])],
                posteriors[self.situations.index(sub_hyper[1])],
            ) for sub_hyper in hyperstage_combinations]

            local_score = max(newscores_list)

            if local_score > 0:
                local_merged = hyperstage_combinations[
                    newscores_list.index(local_score)
                ]
                merge_situ_1, merge_situ_2 = local_merged
                merge_situ_1_idx = self.situations.index(merge_situ_1)
                merge_situ_2_idx = self.situations.index(merge_situ_2)
                merged_situation_list.append(local_merged)

                priors[merge_situ_1_idx] = list(
                    map(
                        add,
                        priors[merge_situ_1_idx],
                        priors[merge_situ_2_idx]
                    )
                )
                posteriors[merge_situ_1_idx] = list(
                    map(
                        add,
                        posteriors[merge_situ_1_idx],
                        posteriors[merge_situ_2_idx]
                    )
                )
                priors[merge_situ_2_idx] = (
                    [0] * len(priors[merge_situ_1_idx]))
                posteriors[merge_situ_2_idx] = (
                    [0] * len(posteriors[merge_situ_1_idx]))

                loglikelihood += local_score
            else:
                break
        merged_situation_list = self._sort_list(merged_situation_list)
        mean_posterior_probs = (
            self._calculate_mean_posterior_probs(
                merged_situation_list,
                posteriors
            )
        )
        return mean_posterior_probs, loglikelihood, merged_situation_list

    def _mark_nodes_with_stage_number(self, merged_situations):
        """AHC algorithm creates a list of indexes to the situations list.
        This function takes those indexes and creates a new list which is
        in a string representation of nodes."""
        self._sort_count = 0
        for index, stage in enumerate(merged_situations):
            if len(stage) > 1:
                for node in stage:
                    self.nodes[node]['stage'] = index

    def _generate_colours_for_situations(self, merged_situations, colour_list):
        """Colours each stage of the tree with an individual colour"""
        number_of_stages = len(merged_situations)
        if colour_list is None:
            stage_colours = Util.generate_colours(number_of_stages)
        else:
            stage_colours = colour_list
            if len(colour_list) < number_of_stages:
                logger.warning(
                    "The number of colours is less than the number" +
                    "of stages. Colours will be recycled."
                )
        self._stage_colours = stage_colours
        iter_colour = cycle(stage_colours)
        for node in self.nodes:
            try:
                stage = self.nodes[node]['stage']
                self.nodes[node]['colour'] = next(iter_colour)
            except KeyError:
                self.nodes[node]['colour'] = 'lightgrey'

    def calculate_AHC_transitions(self, prior=None,
                                  alpha=None, hyperstage=None,
                                  colour_list=None):
        '''Bayesian Agglommerative Hierarchical Clustering algorithm
        implementation. It returns a list of lists of the situations which
        have been merged together, the likelihood of the final model and
        the mean posterior conditional probabilities of the stages.

        User can specify a list of colours to be used for stages. Otherwise,
        colours evenly spaced around the colour spectrum are used.'''
        logger.info("\n\n --- Starting AHC Algorithm ---")

        self.__store_params(prior, alpha, hyperstage)

        mean_posterior_probs, loglikelihood, merged_situations = (
            self._execute_AHC())

        self._mark_nodes_with_stage_number(merged_situations)

        self._generate_colours_for_situations(merged_situations, colour_list)

        self.ahc_output = {
            "Merged Situations": merged_situations,
            "Loglikelihood": loglikelihood,
            "Mean Posterior Probabilities": mean_posterior_probs
        }
        return self.ahc_output

    def create_figure(self, filename):
        """Draws the coloured staged tree for the process described by
        the dataset, and saves it to "<filename>.filetype". Supports
        any filetype that graphviz supports. e.g: "event_tree.png" or
        "event_tree.svg" etc.
        """
        try:
            self.ahc_output
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

        except AttributeError:
            logger.error(
                "----- PLEASE RUN AHC ALGORITHM before trying to" +
                " export graph -----"
            )
            return None
