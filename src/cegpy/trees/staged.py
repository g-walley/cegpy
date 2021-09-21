from ..utilities.util import Util
from ..trees.event import EventTree
from fractions import Fraction
from operator import add, sub
from IPython.display import Image
from IPython import get_ipython
import networkx as nx
import scipy.special
import logging

logger = logging.getLogger('pyceg.staged_tree')


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

    def get_prior_as_list(self):
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

    def get_posterior_as_list(self):
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
        # prior = self.prior.copy()
        hyperstage = self.hyperstage.copy()
        # posterior = self.posterior.copy()
        prior_list = self.get_prior_as_list().copy()
        posterior_list = self.get_posterior_as_list().copy()
        loglikelihood = self._calculate_initial_loglikelihood(
            prior_list, posterior_list
        )
        posterior_probs = self.get_posterior_as_list().copy()
        situ = self.situations.copy()
        merged_situation_list = []
        bayesfactor_score = 1

        logger.info(" ----- Starting main loop of AHC algorithm -----")
        logger.info("Prior is length %d" % len(prior_list))
        while bayesfactor_score > 0:
            local = {}

            for sit_1 in range(len(prior_list)):
                if all(items == 0 for items in posterior_list[sit_1]) is False:
                    model1 = [prior_list[sit_1], posterior_list[sit_1]]
                    for sit_2 in range(sit_1+1, len(prior_list)):
                        is_subset = self._check_issubset(
                            [situ[sit_1], situ[sit_2]],
                            hyperstage
                        )

                        any_non_zero = all(
                            items == 0 for items in posterior_list[sit_2]
                        )

                        if is_subset and not any_non_zero:
                            model2 = [prior_list[sit_2], posterior_list[sit_2]]
                            local[(sit_1, sit_2)] = \
                                self._calculate_bayes_factor(
                                    *model1, *model2
                                )

            max_key = max(local, key=local.get)
            bayesfactor_score = local[max_key]

            if local != {} and bayesfactor_score > 0:
                merged_situation_list.append(list(max_key))

                prior_list[max_key[0]] = list(
                    map(
                        add,
                        prior_list[max_key[0]],
                        prior_list[max_key[1]]
                    )
                )
                posterior_list[max_key[0]] = list(
                    map(
                        add,
                        posterior_list[max_key[0]],
                        posterior_list[max_key[1]]
                    )
                )

                prior_list[max_key[1]] = \
                    [0] * len(prior_list[max_key[0]])
                posterior_list[max_key[1]] = \
                    [0] * len(prior_list[max_key[0]])

                posterior_probs[max_key[0]] = posterior_list[max_key[0]]
                posterior_probs[max_key[1]] = posterior_list[max_key[0]]

                loglikelihood += bayesfactor_score

            logger.debug('--Current AHC score: %d' % bayesfactor_score)

        return posterior_probs, loglikelihood, merged_situation_list

    def _mark_nodes_with_stage_number(self, merged_situation_indexes) -> list:
        """AHC algorithm creates a list of indexes to the situations list.
        This function takes those indexes and creates a new list which is
        in a string representation of nodes."""
        self._sort_count = 0
        list_of_merged_situations = self._sort_list(merged_situation_indexes)
        for index, stage in enumerate(list_of_merged_situations):
            for node_number in stage:
                node_name = 's' + str(node_number)
                self.nodes[node_name]['stage'] = index

        return list_of_merged_situations

    def _generate_colours_for_situations(self, merged_situations):
        """Colours each stage of the tree with an individual colour"""
        number_of_stages = len(merged_situations)
        stage_colours = Util.generate_colours(number_of_stages)
        self._stage_colours = stage_colours

        for node in self.nodes:
            try:
                stage = self.nodes[node]['stage']
                self.nodes[node]['colour'] = stage_colours[stage]
            except KeyError:
                self.nodes[node]['colour'] = 'lightgrey'

    def calculate_AHC_transitions(self, prior=None,
                                  alpha=None, hyperstage=None):
        '''Bayesian Agglommerative Hierarchical Clustering algorithm
        implementation. It returns a list of lists of the situations which
        have been merged together, the likelihood of the final model and
        the mean posterior conditional probabilities of the stages.'''
        logger.info("\n\n --- Starting AHC Algorithm ---")

        self.__store_params(prior, alpha, hyperstage)

        posterior_probs, loglikelihood, \
            merged_situation_indexes = self._execute_AHC_algoritm()

        mean_posterior_probs = \
            self._calculate_mean_posterior_probs(posterior_probs)

        merged_situations = \
            self._mark_nodes_with_stage_number(merged_situation_indexes)

        self._generate_colours_for_situations(merged_situations)

        self.ahc_output = {
            "Merged Situations": merged_situations,
            "Loglikelihood": loglikelihood,
            "Mean Posterior Probabilities": mean_posterior_probs
        }
        return self.ahc_output

    def create_figure(self, filename):
        """Draws the event tree for the process described by the dataset,
        and saves it to <filename>.png"""
        try:
            self.ahc_output
            filename, filetype = Util.generate_filename_and_mkdir(filename)
            logger.info("--- generating graph ---")
            graph = self._generate_pdp_graph()
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
