from src.cegpy import StagedTree
import pandas as pd
from pathlib import Path
from fractions import Fraction as frac
from pydotplus.graphviz import InvocationException
# import xlsxwriter
import numpy as np


class TestStagedTrees():
    def setup(self):
        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_st = StagedTree(
            dataframe=self.med_df,
            sampling_zero_paths=self.med_s_z_paths
        )

        # non-stratified dataset
        fall_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/Falls_Data.xlsx')
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_st = StagedTree(
            dataframe=self.fall_df,
            sampling_zero_paths=self.fall_s_z_paths,
        )

    def test_create_default_prior(self) -> None:
        # stratified medical dataset
        # Expected prior calculated by hand for default alpha of 3
        med_expected_prior = [
            [frac(3, 2), frac(3, 2)],
            [frac(1, 2), frac(1, 2), frac(1, 2)],
            [frac(1, 2), frac(1, 2), frac(1, 2)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
        ]

        fall_expected_prior = [
            [frac(1, 1), frac(1, 1), frac(1, 1), frac(1, 1)],
            [frac(1, 2), frac(1, 2)],
            [frac(1, 2), frac(1, 2)],
            [frac(1, 2), frac(1, 2)],
            [frac(1, 2), frac(1, 2)],
            [frac(1, 6), frac(1, 6), frac(1, 6)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 6), frac(1, 6), frac(1, 6)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 4), frac(1, 4)],
            [frac(1, 12), frac(1, 12)],
            [frac(1, 12), frac(1, 12)],
            [frac(1, 12), frac(1, 12)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 12), frac(1, 12)],
            [frac(1, 12), frac(1, 12)],
            [frac(1, 12), frac(1, 12)],
            [frac(1, 8), frac(1, 8)],
            [frac(1, 8), frac(1, 8)]
        ]
        assert len(med_expected_prior) == 21
        assert len(fall_expected_prior) == 23

        alpha = 3
        # check that prior is correct.
        prior = self.med_st._StagedTree__create_default_prior(alpha)
        assert med_expected_prior == prior

        alpha = 4
        prior = self.fall_st._StagedTree__create_default_prior(alpha)
        assert fall_expected_prior == prior

    def test_create_default_hyperstage(self) -> None:
        med_expected_hyperstage = [
            ["s0", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16",
             "s17", "s18", "s19", "s20"],
            ["s1", "s2"],
            ["s3", "s4", "s5", "s6", "s7", "s8"]
        ]
        fall_expected_hyperstage = [
            ["s0"],
            ["s1", "s2", "s3", "s4"],
            ["s5", "s9"],
            ["s6", "s10"],
            ["s7", "s8", "s11", "s12", "s13", "s14", "s15", "s16", "s17",
             "s22", "s23", "s24", "s25", "s26"]
        ]
        med_hyperstage = self.med_st._StagedTree__create_default_hyperstage()
        print(med_expected_hyperstage)
        print(med_hyperstage)

        fall_hyperstage = self.fall_st._StagedTree__create_default_hyperstage()
        print(fall_expected_hyperstage)
        print(fall_hyperstage)
        assert med_hyperstage == med_expected_hyperstage
        assert fall_hyperstage == fall_expected_hyperstage

    def test_calculate_posterior(self) -> None:
        def calculate_posterior(staged_tree: StagedTree, expected_countset,
                                alpha, expected_likelihood):
            actual_countset = staged_tree._StagedTree__create_edge_countset()
            assert actual_countset == expected_countset

            prior = staged_tree._StagedTree__create_default_prior(alpha)
            staged_tree.prior = prior
            expected_posterior = []
            for idx, countset in enumerate(actual_countset):
                p_elem = []
                for jdx, count in enumerate(countset):
                    p_elem.append(count + prior[idx][jdx])

                expected_posterior.append(p_elem)

            actual_posterior = staged_tree.posterior_list
            assert actual_posterior == expected_posterior
            actual_likelihood = staged_tree._calculate_initial_loglikelihood(
                staged_tree.prior_list,
                staged_tree.posterior_list,
            )
            actual_likelihood = round(actual_likelihood, 2)
            assert actual_likelihood == expected_likelihood

        falls_expected_edge_countset = [
            [379, 1539, 2871, 45211],
            [235, 144],
            [436, 1103],
            [1449, 1422],
            [5375, 39836],
            [53, 52, 130],
            [130, 14],
            [82, 354],
            [870, 233],
            [319, 315, 815],
            [1266, 156],
            [1103, 4272],
            [30769, 9067],
            [11, 42],
            [28, 24],
            [58, 72],
            [102, 28],
            [12, 2],
            [65, 254],
            [151, 164],
            [379, 436],
            [974, 292],
            [133, 23]
        ]
        alpha = 4
        expected_likelihood = -68721.50  # Calculated manually
        assert len(falls_expected_edge_countset) == 23
        calculate_posterior(
            self.fall_st,
            falls_expected_edge_countset,
            alpha,
            expected_likelihood
        )

        med_expected_edge_countset = [
            [5491, 5493],
            [798, 1000, 3693],
            [799, 998, 3696],
            [399, 399],
            [500, 500],
            [1846, 1847],
            [400, 399],
            [500, 498],
            [1849, 1847],
            [393, 6],
            [347, 52],
            [480, 20],
            [433, 67],
            [1482, 364],
            [1305, 542],
            [5, 395],
            [92, 307],
            [70, 430],
            [202, 296],
            [338, 1511],
            [716, 1131]
        ]
        alpha = 3
        expected_likelihood = -30134.07
        assert len(med_expected_edge_countset) == 21
        calculate_posterior(
            self.med_st,
            med_expected_edge_countset,
            alpha,
            expected_likelihood
        )

    def test_merged_leaves_med(self) -> None:
        # check that no leaves have been merged
        self.med_st.calculate_AHC_transitions()
        for stage in self.med_st.ahc_output['Merged Situations']:
            for situ in stage:
                assert situ not in self.med_st.leaves

    def test_merged_leaves_fall(self) -> None:
        self.fall_st.calculate_AHC_transitions()
        for stage in self.fall_st.ahc_output['Merged Situations']:
            for situ in stage:
                assert situ not in self.fall_st.leaves

    def test_independent_hyperstage_generator(self) -> None:
        """generator correctly establishes which subsets have
        cross-over, and returns correct number of subsets"""
        example_hyperstage = [
            ['s1', 's2', 's3'],
            ['s3', 's4'],
            ['s5', 's6'],
            ['s7', 's8'],
            ['s6', 's4']]
        first_hyperstage = set()
        first_hyperstage.add(frozenset(example_hyperstage[0]))
        first_hyperstage.add(frozenset(example_hyperstage[1]))
        first_hyperstage.add(frozenset(example_hyperstage[2]))
        first_hyperstage.add(frozenset(example_hyperstage[4]))
        second_hyperstage = set()
        second_hyperstage.add(frozenset(example_hyperstage[3]))
        expected_hyperstages = set()
        expected_hyperstages.add(frozenset(first_hyperstage))
        expected_hyperstages.add(frozenset(second_hyperstage))

        actual_hyperstages = set()
        for hyperstage in self.fall_st._independent_hyperstage_generator(
                hyperstage=example_hyperstage):
            actual_hyperstages.add(
                frozenset([frozenset(sublist) for sublist in hyperstage]))

        assert actual_hyperstages == expected_hyperstages


class TestChangingDataFrame():
    def setup(self):
        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_st = StagedTree(
            dataframe=self.med_df,
            sampling_zero_paths=self.med_s_z_paths
        )
        self.med_st.calculate_AHC_transitions()
        # non-stratified dataset
        fall_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/Falls_Data.xlsx')
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_st = StagedTree(
            dataframe=self.fall_df,
            sampling_zero_path=self.fall_s_z_paths
        )
        self.fall_st.calculate_AHC_transitions()

    def test_add_empty_column(self) -> None:
        # adding empty column
        med_empty_column_df = self.med_df
        med_empty_column_df["extra"] = ""
        med_empty_column_st = StagedTree(
            dataframe=med_empty_column_df
        )
        med_empty_column_st.calculate_AHC_transitions()
        assert med_empty_column_st.ahc_output == self.med_st.ahc_output

        fall_empty_column_df = self.fall_df
        fall_empty_column_df["extra"] = ""
        fall_empty_column_st = StagedTree(
            dataframe=fall_empty_column_df
        )
        fall_empty_column_st.calculate_AHC_transitions()
        assert fall_empty_column_st.ahc_output == self.fall_st.ahc_output

    def test_add_NA_column(self) -> None:
        # adding NA column
        med_add_NA_df = self.med_df
        med_add_NA_df["extra"] = np.nan
        med_add_NA_st = StagedTree(
            dataframe=med_add_NA_df
        )
        med_add_NA_st.calculate_AHC_transitions()
        assert med_add_NA_st.ahc_output == self.med_st.ahc_output

        fall_add_NA_df = self.fall_df
        fall_add_NA_df["extra"] = np.nan
        fall_add_NA_st = StagedTree(
            dataframe=fall_add_NA_df
        )
        fall_add_NA_st.calculate_AHC_transitions()
        assert fall_add_NA_st.ahc_output == self.fall_st.ahc_output

    def test_add_same_column_med(self) -> None:
        # adding column with no more information
        med_add_same_df = self.med_df
        med_add_same_df["extra"] = "same for all"
        med_add_same_st = StagedTree(
            dataframe=med_add_same_df
        )
        med_add_same_st.calculate_AHC_transitions()
        try:
            med_add_same_st.create_figure(
                "out/test_add_same_column_med_fig.pdf")
        except InvocationException:
            pass

        first_set = set(
            tuple(x) for x in self.med_st.ahc_output['Merged Situations']
        )
        second_set = set(
            tuple(x) for x in med_add_same_st.ahc_output['Merged Situations']
        )
        assert first_set.issubset(second_set)

    def test_add_same_column_fall(self) -> None:
        fall_add_same_df = self.fall_df
        fall_add_same_df["extra"] = "same for all"
        fall_add_same_st = StagedTree(
            dataframe=fall_add_same_df
        )
        fall_add_same_st.calculate_AHC_transitions()
        try:
            fall_add_same_st.create_figure(
                "out/test_add_same_column_fall_fig.pdf")
        except InvocationException:
            pass

        first_set = set(
            tuple(x) for x in self.fall_st.ahc_output['Merged Situations']
        )
        second_set = set(
            tuple(x) for x in fall_add_same_st.ahc_output['Merged Situations']
        )
        assert first_set.issubset(second_set)

    def test_add_same_column_int_med(self) -> None:
        # adding column with no more information
        med_add_same_df = self.med_df
        med_add_same_df["extra"] = 1
        med_add_same_st = StagedTree(
            dataframe=med_add_same_df
        )
        med_add_same_st.calculate_AHC_transitions()
        try:
            med_add_same_st.create_figure(
                "out/test_add_same_column_int_med_fig.pdf")
        except InvocationException:
            pass

        first_set = set(
            tuple(x) for x in self.med_st.ahc_output['Merged Situations']
        )
        second_set = set(
            tuple(x) for x in med_add_same_st.ahc_output['Merged Situations']
        )
        assert first_set.issubset(second_set)

    def test_add_same_column_int_fall(self) -> None:
        fall_add_same_df = self.fall_df
        fall_add_same_df["extra"] = 1
        fall_add_same_st = StagedTree(
            dataframe=fall_add_same_df
        )
        fall_add_same_st.calculate_AHC_transitions()
        try:
            fall_add_same_st.create_figure(
                "out/test_add_same_column_int_fall_fig.pdf")
        except InvocationException:
            pass

        first_set = set(
            tuple(x) for x in self.fall_st.ahc_output['Merged Situations']
        )
        second_set = set(
            tuple(x) for x in fall_add_same_st.ahc_output['Merged Situations']
        )
        assert first_set.issubset(second_set)


class TestWithDynamicDataset():
    def setup(self):
        self.df = pd.read_excel("data/Falls_Dynamic_Data.xlsx")

    def test_single_floret_stages(self):
        """Single floret stages are marked as same stage"""
        df = self.df[["Residence", "Risk", "Treatment", "Fall", "Outcome"]]
        st = StagedTree(df)
        ahc_out = st.calculate_AHC_transitions()
        ms = [set(merged_sits) for merged_sits in ahc_out["Merged Situations"]]
        assert {"s9", "s13", "s14"} in ms
