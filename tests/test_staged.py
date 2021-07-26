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
        assert isinstance(self.med_st.get_node_list(), list)
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
        assert len(med_expected_prior) == 21
        alpha = 3
        # check that prior is correct.
        prior = self.med_st._generate_default_prior(alpha)
        assert med_expected_prior == prior

        # for node_idx, node_priors in enumerate(prior):
        #     # are elements (node_priors) of prior a list?
        #     assert isinstance(node_priors, list)
        #     # correct number priors in each sub-list (i)?
        #     assert len(node_priors) == len(med_expected_prior[node_idx])
        #     for count_idx, count in enumerate(node_priors):
        #         # count type is a frac?
        #         assert isinstance(count, frac)
        #         # check node_prior count is same as in expected prior
        #         assert count == med_expected_prior[node_idx][count_idx]
