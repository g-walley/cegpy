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
        prior = self.med_st._create_default_prior(alpha)
        assert med_expected_prior == prior

        alpha = 4
        prior = self.fall_st._create_default_prior(alpha)
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
        med_hyperstage = self.med_st._create_default_hyperstage()
        print(med_expected_hyperstage)
        print(med_hyperstage)

        fall_hyperstage = self.fall_st._create_default_hyperstage()
        print(fall_expected_hyperstage)
        print(fall_hyperstage)
        assert med_hyperstage == med_expected_hyperstage
        assert fall_hyperstage == fall_expected_hyperstage

    def test_create_edge_countset(self) -> None:
        med_expected_edge_countset = [
            [5500, 5500],
            [800, 1000, 3700],
            [800, 1000, 3700],
            [400, 400],
            [500, 500],
            [1850, 1850],
            [400, 400],
            [500, 500],
            [1850, 1850],
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
        assert len(med_expected_edge_countset) == 21
        assert len(falls_expected_edge_countset) == 23

        actual_countset = self.med_st._create_edge_countset()
        assert actual_countset == med_expected_edge_countset

        actual_countset = self.fall_st._create_edge_countset()
        assert actual_countset == falls_expected_edge_countset
