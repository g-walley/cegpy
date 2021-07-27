from ..src.cegpy.trees.staged import StagedTree
import pandas as pd
from pathlib import Path
from fractions import Fraction as frac


class TestStagedTrees():
    def setup(self):
        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_st = StagedTree(
            {
                'dataframe': self.med_df,
                'sampling_zero_paths': self.med_s_z_paths
            }
        )

        # non-stratified dataset
        fall_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/Falls_Data.xlsx')
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_st = StagedTree(
            {
                'dataframe': self.fall_df,
                'sampling_zero_paths': self.fall_s_z_paths,
            }
        )

    def test_event_tree_calls(self) -> None:
        assert isinstance(self.med_st.get_edges(), list)
        assert isinstance(self.med_st.get_nodes(), list)
        assert isinstance(self.med_st.get_sampling_zero_paths(),
                          type(self.med_s_z_paths))
        assert isinstance(self.med_st.get_edge_labels(), list)

    def test_generate_default_prior(self) -> None:
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
        prior = self.med_st._generate_default_prior(alpha)
        assert med_expected_prior == prior

        alpha = 4
        prior = self.fall_st._generate_default_prior(alpha)
        assert fall_expected_prior == prior

    # def test_generate_default_hyperstage(self) -> None:
