from ..src.cegpy.trees.event import EventTree
from collections import defaultdict
import pandas as pd
# from ceg_util import CegUtil as util
from pathlib import Path


class TestEventTree():
    def setup(self):
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')

        self.df = pd.read_excel(df_path)
        self.et = EventTree({'dataframe': self.df})

        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_st = EventTree(
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
        self.fall_st = EventTree(
            {
                'dataframe': self.fall_df,
                'sampling_zero_paths': self.fall_s_z_paths,
            }
        )

    def test_check_sampling_zero_paths_param(self) -> None:
        """Tests the function that is checking the sampling zero paths param"""
        szp = [('Medium',), ('Medium', 'High')]
        assert self.et._check_sampling_zero_paths_param(szp) == szp

        szp = [1, 2, 3, 4]
        assert self.et._check_sampling_zero_paths_param(szp) is None

        szp = [('path', 'to'), (123, 'something'), 'path/to']
        assert self.et._check_sampling_zero_paths_param(szp) is None

    def test_check_sampling_zero_get_and_set(self) -> None:
        """Tests the functions that set and get the sampling zeros"""
        assert self.et.get_sampling_zero_paths() is None

        szp = [('Medium',), ('Medium', 'High')]
        self.et._set_sampling_zero_paths(szp)
        assert self.et.get_sampling_zero_paths() == szp

    def test_create_node_list_from_paths(self) -> None:
        paths = defaultdict(int)
        paths[('path',)] += 1
        paths[('path', 'to')] += 1
        paths[('path', 'away')] += 1
        paths[('road',)] += 1
        paths[('road', 'to')] += 1
        paths[('road', 'away')] += 1

        # code being tested:
        node_list = self.et._create_node_list_from_paths(paths)

        print(node_list)
        assert len(list(paths.keys())) + 1 == len(node_list)
        assert node_list[0] == 's0'
        assert node_list[-1] == 's%d' % (len(node_list) - 1)

    def test_construct_event_tree(self) -> None:
        """Tests the construction of an event tree from a set of paths,
        nodes, and """
        event_tree = self.et._construct_event_tree()
        assert isinstance(event_tree, defaultdict)
        for key, value in event_tree.items():
            assert isinstance(key, tuple)
            for elem in key:
                assert isinstance(elem, tuple)

                for sub_elem in elem[0]:
                    assert isinstance(sub_elem, str)

            # Check the edge data in the key is exactly 2 long.
            assert len(key[1]) == 2

    def test_get_functions_producing_expected_data(self) -> None:
        edge_labels = self.et.get_edge_labels()
        assert isinstance(edge_labels, list)
        for path in edge_labels:
            assert isinstance(path, tuple)
            for edge_label in path:
                assert isinstance(edge_label, str)

        edges = self.et.get_edges()
        assert isinstance(edges, list)
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert isinstance(edge[0], str)
            assert isinstance(edge[1], str)

        nodes = self.et.get_nodes()
        check_list_contains_strings(nodes)

        situations = self.et.get_situations()
        check_list_contains_strings(situations)

        leaves = self.et.get_leaves()
        check_list_contains_strings(leaves)

        emanating_nodes = self.et.get_emanating_nodes()
        check_list_contains_strings(emanating_nodes)

        terminating_nodes = self.et.get_terminating_nodes()
        check_list_contains_strings(terminating_nodes)

    def test_get_categories_per_variable(self) -> None:
        expected_med_cats_per_var = {
            "Classification": 2,
            "Group": 3,
            "Difficulty": 2,
            "Response": 2,
        }
        med_cats_per_var = self.med_st.get_categories_per_variable()
        assert expected_med_cats_per_var == med_cats_per_var

        expected_fall_cats_per_var = {
            "HousingAssessment": 4,
            "Risk": 2,
            "Treatment": 3,
            "Fall": 2,
        }
        fall_cats_per_var = self.fall_st.get_categories_per_variable()
        assert expected_fall_cats_per_var == fall_cats_per_var


def check_list_contains_strings(str_list) -> bool:
    assert isinstance(str_list, list)
    for elem in str_list:
        assert isinstance(elem, str)
