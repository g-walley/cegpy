"""Tests StagedTree class."""
# pylint: disable=too-many-lines,protected-access
from pathlib import Path
import unittest
from fractions import Fraction
import pytest
import pandas as pd
from pydotplus.graphviz import InvocationException
import numpy as np

# import xlsxwriter
from cegpy import StagedTree
from cegpy.trees._staged import _calculate_mean_posterior_probs


class TestLogging:
    """Tests logging in stagedtree"""

    def setup(self):
        """Test setup"""
        # pylint: disable=attribute-defined-outside-init
        # stratified dataset
        med_df_path = (
            Path(__file__)
            .resolve()
            .parent.parent.joinpath("../data/medical_dm_modified.xlsx")
        )
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_st = StagedTree(
            dataframe=self.med_df, sampling_zero_paths=self.med_s_z_paths
        )

        # non-stratified dataset
        fall_df_path = (
            Path(__file__).resolve().parent.parent.joinpath("../data/Falls_Data.xlsx")
        )
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_st = StagedTree(
            dataframe=self.fall_df,
            sampling_zero_paths=self.fall_s_z_paths,
        )

    def test_figure_with_wrong_edge_attribute(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Ensures a warning is raised when a non-existent
        attribute is passed for the edge_info argument"""
        msg = (
            r"edge_info 'prob' does not exist for the "
            r"StagedTree class. Using the default of 'count' values "
            r"on edges instead. For more information, see the "
            r"documentation."
        )

        # stratified medical dataset
        self.med_st.calculate_AHC_transitions()
        _ = self.med_st.create_figure(filename=None, edge_info="prob")
        assert msg in caplog.text, "Expected log message not logged."

        # non-stratified dataset
        self.fall_st.calculate_AHC_transitions()
        _ = self.fall_st.create_figure(filename=None, edge_info="prob")
        assert msg in caplog.text, "Expected log message not logged."

    def test_run_ahc_before_figure(self, caplog) -> None:
        """Tests expected error message is in the log when running without
        running AHC"""
        try:
            self.med_st.create_figure()
            assert "PLEASE RUN AHC" in caplog.text
        except InvocationException:
            pass


class TestStagedTrees(unittest.TestCase):
    """Tests for staged trees with a stratified and non stratified dataset"""

    def setUp(self):
        # stratified dataset
        med_df_path = (
            Path(__file__)
            .resolve()
            .parent.parent.joinpath("../data/medical_dm_modified.xlsx")
        )
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_st = StagedTree(
            dataframe=self.med_df, sampling_zero_paths=self.med_s_z_paths
        )

        # non-stratified dataset
        fall_df_path = (
            Path(__file__).resolve().parent.parent.joinpath("../data/Falls_Data.xlsx")
        )
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_st = StagedTree(
            dataframe=self.fall_df,
            sampling_zero_paths=self.fall_s_z_paths,
        )

    def test_check_hyperstage(self) -> None:
        # stratified medical dataset
        med_hyperstage_less = [
            [
                "s0",
                "s9",
                "s10",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s19",
                "s20",
            ],
            ["s1", "s2"],
            ["s3", "s4", "s6", "s7", "s8"],
        ]

        med_hyperstage_more = [
            [
                "s0",
                "s9",
                "s10",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s18",
                "s19",
                "s20",
            ],
            ["s1", "s2", "s50"],
            ["s3", "s4", "s5", "s6", "s7", "s8"],
        ]

        med_hyperstage_unequal = [
            [
                "s0",
                "s1",
                "s9",
                "s10",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s18",
                "s19",
                "s20",
            ],
            ["s2", "s4", "s50"],
            ["s3", "s5", "s6", "s7", "s8"],
        ]

        with pytest.raises(ValueError):
            self.med_st._check_hyperstages(med_hyperstage_less)

        with pytest.raises(ValueError):
            self.med_st._check_hyperstages(med_hyperstage_more)

        with pytest.raises(ValueError):
            self.med_st._check_hyperstages(med_hyperstage_unequal)

        # non-stratified falls dataset
        fall_hyperstage_less = [
            ["s1", "s2", "s3", "s4"],
            ["s5", "s9"],
            ["s6", "s10"],
            [
                "s7",
                "s8",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s22",
                "s23",
                "s24",
                "s25",
                "s26",
            ],
        ]

        fall_hyperstage_more = [
            ["s0", "v10"],
            ["s1", "s2", "s3", "s4"],
            ["s5", "s9", "pg"],
            ["s6", "s10"],
            [
                "s7",
                "s8",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s22",
                "s23",
                "s24",
                "s25",
                "s26",
            ],
        ]

        fall_hyperstage_unequal = [
            ["s0", "s2"],
            ["s1", "s3", "s4"],
            ["s5", "s9", "s8"],
            ["s6", "s10"],
            [
                "s7",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s22",
                "s23",
                "s24",
                "s25",
                "s26",
            ],
        ]

        with pytest.raises(ValueError):
            self.fall_st._check_hyperstages(fall_hyperstage_less)

        with pytest.raises(ValueError):
            self.fall_st._check_hyperstages(fall_hyperstage_more)

        with pytest.raises(ValueError):
            self.fall_st._check_hyperstages(fall_hyperstage_unequal)

    def test_check_prior(self) -> None:
        # stratified medical dataset
        med_prior_incorrect_length = [
            [Fraction(3, 2), Fraction(3, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
        ]

        med_prior_unequal = [
            [Fraction(3, 2), Fraction(3, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
        ]

        med_prior_negative = [
            [Fraction(-3, 2), Fraction(3, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 4), Fraction(-1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(-1, 8), Fraction(-1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
        ]

        with pytest.raises(ValueError):
            self.med_st._check_prior(med_prior_incorrect_length)

        with pytest.raises(ValueError):
            self.med_st._check_prior(med_prior_unequal)

        with pytest.raises(ValueError):
            self.med_st._check_prior(med_prior_negative)

        # non-stratified falls dataset
        fall_prior_incorrect_length = [
            [Fraction(1, 1), Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 6), Fraction(1, 6), Fraction(1, 6)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
        ]

        fall_prior_unequal = [
            [Fraction(1, 1), Fraction(1, 1)],
            [Fraction(1, 2), Fraction(1, 1), Fraction(1, 1), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 6), Fraction(1, 6), Fraction(1, 6)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 6), Fraction(1, 6), Fraction(1, 6)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8), Fraction(1, 8)],
        ]

        fall_prior_negative = [
            [Fraction(1, 1), Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 6), Fraction(-1, 6), Fraction(1, 6)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(-1, 4)],
            [Fraction(1, 6), Fraction(1, 6), Fraction(1, 6)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 12), Fraction(-1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
        ]

        with pytest.raises(ValueError):
            self.fall_st._check_prior(fall_prior_incorrect_length)

        with pytest.raises(ValueError):
            self.fall_st._check_prior(fall_prior_unequal)

        with pytest.raises(ValueError):
            self.fall_st._check_prior(fall_prior_negative)

    def test_check_prior_assigned(self) -> None:
        # stratified medical dataset
        med_prior_noerror = [
            [Fraction(34, 2), Fraction(3, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(51, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(11, 42)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(21, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
        ]
        self.med_st.calculate_AHC_transitions(prior=med_prior_noerror)
        assert self.med_st.prior_list == med_prior_noerror

        # non-stratified falls dataset
        fall_prior_noerror = [
            [Fraction(1, 1), Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(21, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 6), Fraction(1, 6), Fraction(1, 6)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(31, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 6), Fraction(15, 6), Fraction(1, 6)],
            [Fraction(12, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
        ]

        self.fall_st.calculate_AHC_transitions(prior=fall_prior_noerror)
        assert self.fall_st.prior_list == fall_prior_noerror

    def test_check_hyperstage_assigned(self) -> None:
        # stratified medical dataset
        med_hyperstage_noerror = [
            ["s0", "s9", "s10", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20"],
            ["s1", "s2"],
            ["s3", "s4", "s5", "s11", "s12", "s6", "s7", "s8"],
        ]

        self.med_st.calculate_AHC_transitions(hyperstage=med_hyperstage_noerror)
        assert self.med_st.hyperstage == med_hyperstage_noerror

        # non-stratified falls dataset
        fall_hyperstage_noerror = [
            ["s0"],
            ["s2", "s3", "s4"],
            ["s5", "s9"],
            ["s1", "s6", "s10"],
            [
                "s7",
                "s8",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s22",
                "s23",
                "s24",
                "s25",
                "s26",
            ],
        ]

        self.fall_st.calculate_AHC_transitions(hyperstage=fall_hyperstage_noerror)
        assert self.fall_st.hyperstage == fall_hyperstage_noerror

    def test_hyperstage_noerror(self) -> None:
        # stratified medical dataset
        med_hyperstage_noerror = [
            ["s0", "s9", "s10", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20"],
            ["s1", "s2"],
            ["s3", "s4", "s5", "s11", "s12", "s6", "s7", "s8"],
        ]

        try:
            self.med_st._check_hyperstages(med_hyperstage_noerror)
        except ValueError as exc:
            self.fail(f"'check_hyperstages' raised an exception {exc}")

        # non-stratified falls dataset
        fall_hyperstage_noerror = [
            ["s0"],
            ["s2", "s3", "s4"],
            ["s5", "s9"],
            ["s1", "s6", "s10"],
            [
                "s7",
                "s8",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s22",
                "s23",
                "s24",
                "s25",
                "s26",
            ],
        ]

        try:
            self.fall_st._check_hyperstages(fall_hyperstage_noerror)
        except ValueError as exc:
            self.fail(f"'check_hyperstages' raised an exception {exc}")

    def test_create_default_prior(self) -> None:
        # stratified medical dataset
        # Expected prior calculated by hand for default alpha of 3
        med_expected_prior = [
            [Fraction(3, 2), Fraction(3, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
        ]

        fall_expected_prior = [
            [Fraction(1, 1), Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 6), Fraction(1, 6), Fraction(1, 6)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 6), Fraction(1, 6), Fraction(1, 6)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 12), Fraction(1, 12)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
        ]
        assert len(med_expected_prior) == 21
        assert len(fall_expected_prior) == 23

        alpha = 3
        # check that prior is correct.
        prior = self.med_st._create_default_prior(alpha)
        assert med_expected_prior == prior

        alpha = 4
        prior = self.fall_st._create_default_prior(alpha)
        assert fall_expected_prior == prior

    def test_create_default_hyperstage(self) -> None:
        med_expected_hyperstage = [
            [
                "s0",
                "s9",
                "s10",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s18",
                "s19",
                "s20",
            ],
            ["s1", "s2"],
            ["s3", "s4", "s5", "s6", "s7", "s8"],
        ]
        fall_expected_hyperstage = [
            ["s0"],
            ["s1", "s2", "s3", "s4"],
            ["s5", "s9"],
            ["s6", "s10"],
            [
                "s7",
                "s8",
                "s11",
                "s12",
                "s13",
                "s14",
                "s15",
                "s16",
                "s17",
                "s22",
                "s23",
                "s24",
                "s25",
                "s26",
            ],
        ]
        med_hyperstage = self.med_st._create_default_hyperstage()

        fall_hyperstage = self.fall_st._create_default_hyperstage()
        assert med_hyperstage == med_expected_hyperstage
        assert fall_hyperstage == fall_expected_hyperstage

    def test_calculate_posterior(self) -> None:
        def calculate_posterior(
            staged_tree: StagedTree, expected_countset, alpha, expected_likelihood
        ):
            actual_countset = staged_tree._create_edge_countset()
            assert actual_countset == expected_countset

            prior = staged_tree._create_default_prior(alpha)
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
            [133, 23],
        ]
        alpha = 4
        expected_likelihood = -68721.50  # Calculated manually
        assert len(falls_expected_edge_countset) == 23
        calculate_posterior(
            self.fall_st, falls_expected_edge_countset, alpha, expected_likelihood
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
            [716, 1131],
        ]
        alpha = float(3)
        expected_likelihood = -30134.07
        assert len(med_expected_edge_countset) == 21
        calculate_posterior(
            self.med_st, med_expected_edge_countset, alpha, expected_likelihood
        )

    def test_prior_alpha_conflict(self) -> None:
        prior = [
            [Fraction(3, 2), Fraction(3, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
            [Fraction(1, 8), Fraction(1, 8)],
        ]
        alpha = 4
        self.med_st.calculate_AHC_transitions(prior=prior, alpha=alpha)
        assert self.med_st.alpha is None

    def test_alpha_format(self) -> None:
        with pytest.raises(TypeError):
            self.med_st.calculate_AHC_transitions(alpha={"5"})

    def test_merged_leaves_med(self) -> None:
        # check that no leaves have been merged
        self.med_st.calculate_AHC_transitions()
        for stage in self.med_st.ahc_output["Merged Situations"]:
            for situ in stage:
                assert situ not in self.med_st.leaves

    def test_merged_leaves_fall(self) -> None:
        self.fall_st.calculate_AHC_transitions()
        for stage in self.fall_st.ahc_output["Merged Situations"]:
            for situ in stage:
                assert situ not in self.fall_st.leaves

    def test_independent_hyperstage_generator(self) -> None:
        """generator correctly establishes which subsets have
        cross-over, and returns correct number of subsets"""
        example_hyperstage = [
            ["s1", "s2", "s3"],
            ["s3", "s4"],
            ["s5", "s6"],
            ["s7", "s8"],
            ["s6", "s4"],
        ]
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
            hyperstages=example_hyperstage
        ):
            actual_hyperstages.add(
                frozenset([frozenset(sublist) for sublist in hyperstage])
            )

        assert actual_hyperstages == expected_hyperstages

    def test_node_colours(self) -> None:
        """Ensures that all nodes in the event tree dot graph object
        are coloured in lightgrey and the nodes in the staged tree
        agree with the result of AHC"""
        dot_event_nodes = self.med_st.dot_event_graph().get_nodes()
        self.med_st.calculate_AHC_transitions()
        dot_staged_nodes = self.med_st.dot_staged_graph().get_nodes()
        event_node_colours = [
            n.obj_dict["attributes"]["fillcolor"] for n in dot_event_nodes
        ]
        staged_node_colours = [
            n.obj_dict["attributes"]["fillcolor"] for n in dot_staged_nodes
        ]
        assert len(set(event_node_colours)) == 1
        assert event_node_colours[0] == "lightgrey"
        assert len(set(staged_node_colours)) > 1
        g = self.med_st.dot_staged_graph()
        for node in self.med_st.nodes():
            assert (
                g.get_node(node)[0].obj_dict["attributes"]["fillcolor"]
                == self.med_st.nodes[node]["colour"]
            )

    def test_new_colours(self) -> None:
        colours = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462"]
        self.fall_st.calculate_AHC_transitions(colour_list=colours)
        dot_staged_nodes = self.fall_st.dot_staged_graph().get_nodes()
        staged_node_colours = [
            n.obj_dict["attributes"]["fillcolor"] for n in dot_staged_nodes
        ]
        assert set(colours + ["lightgrey"]) == set(staged_node_colours)

    def test_new_colours_length(self) -> None:
        colours = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072"]
        with pytest.raises(IndexError):
            self.fall_st.calculate_AHC_transitions(colour_list=colours)


class TestChangingDataFrame(unittest.TestCase):
    """Tests changing DataFrame"""

    def setUp(self):
        # stratified dataset
        med_df_path = (
            Path(__file__)
            .resolve()
            .parent.parent.joinpath("../data/medical_dm_modified.xlsx")
        )
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_st = StagedTree(
            dataframe=self.med_df, sampling_zero_paths=self.med_s_z_paths
        )
        self.med_st.calculate_AHC_transitions()
        # non-stratified dataset
        fall_df_path = (
            Path(__file__).resolve().parent.parent.joinpath("../data/Falls_Data.xlsx")
        )
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_st = StagedTree(
            dataframe=self.fall_df, sampling_zero_path=self.fall_s_z_paths
        )
        self.fall_st.calculate_AHC_transitions()

    def test_add_empty_column(self) -> None:
        # adding empty column
        med_empty_column_df = self.med_df
        med_empty_column_df["extra"] = ""
        med_empty_column_st = StagedTree(dataframe=med_empty_column_df)
        med_empty_column_st.calculate_AHC_transitions()
        assert med_empty_column_st.ahc_output == self.med_st.ahc_output

        fall_empty_column_df = self.fall_df
        fall_empty_column_df["extra"] = ""
        fall_empty_column_st = StagedTree(dataframe=fall_empty_column_df)
        fall_empty_column_st.calculate_AHC_transitions()
        assert fall_empty_column_st.ahc_output == self.fall_st.ahc_output

    def test_add_NA_column(self) -> None:
        # adding NA column
        med_add_NA_df = self.med_df
        med_add_NA_df["extra"] = np.nan
        med_add_NA_st = StagedTree(dataframe=med_add_NA_df)
        med_add_NA_st.calculate_AHC_transitions()
        assert med_add_NA_st.ahc_output == self.med_st.ahc_output

        fall_add_NA_df = self.fall_df
        fall_add_NA_df["extra"] = np.nan
        fall_add_NA_st = StagedTree(dataframe=fall_add_NA_df)
        fall_add_NA_st.calculate_AHC_transitions()
        assert fall_add_NA_st.ahc_output == self.fall_st.ahc_output

    def test_add_same_column_med(self) -> None:
        # adding column with no more information
        med_add_same_df = self.med_df
        med_add_same_df["extra"] = "same for all"
        med_add_same_st = StagedTree(dataframe=med_add_same_df)
        med_add_same_st.calculate_AHC_transitions()
        try:
            med_add_same_st.create_figure("out/test_add_same_column_med_fig.pdf")
        except InvocationException:
            pass

        first_set = set(tuple(x) for x in self.med_st.ahc_output["Merged Situations"])
        second_set = set(
            tuple(x) for x in med_add_same_st.ahc_output["Merged Situations"]
        )
        assert first_set.issubset(second_set)

    def test_add_same_column_fall(self) -> None:
        fall_add_same_df = self.fall_df
        fall_add_same_df["extra"] = "same for all"
        fall_add_same_st = StagedTree(dataframe=fall_add_same_df)
        fall_add_same_st.calculate_AHC_transitions()
        try:
            fall_add_same_st.create_figure("out/test_add_same_column_fall_fig.pdf")
        except InvocationException:
            pass

        first_set = set(tuple(x) for x in self.fall_st.ahc_output["Merged Situations"])
        second_set = set(
            tuple(x) for x in fall_add_same_st.ahc_output["Merged Situations"]
        )
        assert first_set.issubset(second_set)

    def test_add_same_column_int_med(self) -> None:
        # adding column with no more information
        med_add_same_df = self.med_df
        med_add_same_df["extra"] = 1
        med_add_same_st = StagedTree(dataframe=med_add_same_df)
        med_add_same_st.calculate_AHC_transitions()
        try:
            med_add_same_st.create_figure("out/test_add_same_column_int_med_fig.pdf")
        except InvocationException:
            pass

        first_set = set(tuple(x) for x in self.med_st.ahc_output["Merged Situations"])
        second_set = set(
            tuple(x) for x in med_add_same_st.ahc_output["Merged Situations"]
        )
        assert first_set.issubset(second_set)

    def test_add_same_column_int_fall(self) -> None:
        fall_add_same_df = self.fall_df
        fall_add_same_df["extra"] = 1
        fall_add_same_st = StagedTree(dataframe=fall_add_same_df)
        fall_add_same_st.calculate_AHC_transitions()
        try:
            fall_add_same_st.create_figure("out/test_add_same_column_int_fall_fig.pdf")
        except InvocationException:
            pass

        first_set = set(tuple(x) for x in self.fall_st.ahc_output["Merged Situations"])
        second_set = set(
            tuple(x) for x in fall_add_same_st.ahc_output["Merged Situations"]
        )
        assert first_set.issubset(second_set)


class TestWithDynamicDataset(unittest.TestCase):
    """Dynamic test set test"""

    def setUp(self):
        self.dataframe = pd.read_excel("data/Falls_Dynamic_Data.xlsx")

    def test_single_floret_stages(self):
        """Single floret stages are marked as same stage"""
        dataframe = self.dataframe[
            ["Residence", "Risk", "Treatment", "Fall", "Outcome"]
        ]
        s_tree = StagedTree(dataframe)
        ahc_out = s_tree.calculate_AHC_transitions()
        merged_sits = [set(merged_sits) for merged_sits in ahc_out["Merged Situations"]]
        assert {"s9", "s13", "s14"} in merged_sits


class TestNumericalDataset(unittest.TestCase):
    """Test case for purely numerical dataset."""

    def setUp(self):
        """setup for tests"""
        self.data = pd.read_csv("data/Asym.csv")

    def test_string_missing_paths(self):
        """Missing paths are provided as strings, there's no error."""
        missing_paths = [
            ("0", "1", "1", "1"),
            ("1", "0", "1", "1"),
            ("0", "1", "0", "1"),
        ]
        try:
            _ = StagedTree(self.data, sampling_zero_paths=missing_paths)
        # pylint: disable=broad-except
        except Exception as err:
            self.fail(f"There was an error when using string missing_paths:\n{err}")

    def test_numerical_missing_paths(self):
        """Missing paths are provided as numerical data, there's no error."""
        missing_paths = [(0, 1, 1, 1), (1, 0, 1, 1), (0, 1, 0, 1)]
        try:
            _ = StagedTree(self.data, sampling_zero_paths=missing_paths)
        # pylint: disable=broad-except
        except Exception as err:
            self.fail(f"There was an error when using string missing_paths:\n{err}")


class TestPosteriorProbabilityCalculations(unittest.TestCase):
    """Tests the _calculate_and_apply_mean_posterior_probs() functions"""

    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(
            [
                np.array(["1", "Trt1", "Recover"]),
                np.array(["1", "Trt1", "Dont Recover"]),
                np.array(["2", "Trt1", "Recover"]),
                np.array(["2", "Trt1", "Dont Recover"]),
                np.array(["1", "Trt2", "Recover"]),
                np.array(["1", "Trt2", "Dont Recover"]),
                np.array(["2", "Trt2", "Recover"]),
                np.array(["2", "Trt2", "Dont Recover"]),
            ]
        )
        self.staged = StagedTree(self.dataframe)
        self.merged_situations = [
            ("s0", "s1"),
            ("s3", "s5", "s6"),
            ("s2",),
            ("s4",),
        ]
        self.probs = [
            [50, 70],
            [0],
            [125, 310],
            [0],
            [12, 82],
            [0],
            [55, 352],
        ]

    def test_merged_probabilities(self):
        actual_mpp = _calculate_mean_posterior_probs(
            self.staged.situations, self.merged_situations, self.probs
        )
        expected_mpp = [
            [0.417, 0.583],
            [0.135, 0.865],
            [0.287, 0.713],
            [0.128, 0.872],
        ]
        self.assertEqual(actual_mpp, expected_mpp)

    def test_merged_probabilities_are_applied(self):
        """Merged probabilites are applied to the StagedTree."""
        expected_mpp = [
            [0.417, 0.583],
            [0.135, 0.865],
            [0.287, 0.713],
            [0.128, 0.872],
        ]
        self.staged._apply_mean_posterior_probs(self.merged_situations, expected_mpp)
        edges = {
            ("s0", "s1", "1"): 0.417,
            ("s0", "s2", "2"): 0.583,
            ("s1", "s3", "Trt1"): 0.417,
            ("s1", "s4", "Trt2"): 0.583,
            ("s2", "s5", "Trt1"): 0.287,
            ("s2", "s6", "Trt2"): 0.713,
            ("s3", "s7", "Dont Recover"): 0.135,
            ("s3", "s8", "Recover"): 0.865,
            ("s4", "s9", "Dont Recover"): 0.128,
            ("s4", "s10", "Recover"): 0.872,
            ("s5", "s11", "Dont Recover"): 0.135,
            ("s5", "s12", "Recover"): 0.865,
            ("s6", "s13", "Dont Recover"): 0.135,
            ("s6", "s14", "Recover"): 0.865,
        }
        for edge, probability in edges.items():
            self.assertEqual(self.staged.edges[edge]["probability"], probability)
