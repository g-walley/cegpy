import pandas as pd
import re
import numpy as np
import pytest
from collections import defaultdict
from pathlib import Path
from pytest_mock import MockerFixture
from src.cegpy import EventTree
from pydotplus.graphviz import InvocationException


class TestEventTreeAPI():
    def setup(self):
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.df = pd.read_excel(df_path)

    def test_required_argument_missing_fails(self):
        pytest.raises(TypeError, EventTree)

    def test_required_argument_wrong_type_fails(self):
        dataframe = 5
        pytest.raises(ValueError, EventTree, dataframe=dataframe)

    def test_incorrect_sampling_zero_fails(self):
        szp = [('edge_1'), ('edge_1', 'edge_2')]
        pytest.raises(
            ValueError,
            EventTree,
            dataframe=self.df,
            sampling_zero_paths=szp)


class TestEventTree():
    def setup(self):
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')

        self.df = pd.read_excel(df_path)
        self.et = EventTree(dataframe=self.df)
        self.reordered_et = EventTree(
            dataframe=self.df,
            var_order=self.df.columns[::-1]
        )
        self.node_format = re.compile('^s\\d\\d*$')

    def test_check_sampling_zero_paths_param(self) -> None:
        """Tests the function that is checking the sampling zero paths param"""
        szp = [('Medium',), ('Medium', 'High')]
        assert self.et._EventTree__check_sampling_zero_paths_param(szp) == szp
        szp = [1, 2, 3, 4]
        assert self.et._EventTree__check_sampling_zero_paths_param(szp) is None

        szp = [('path', 'to'), (123, 'something'), 'path/to']
        assert self.et._EventTree__check_sampling_zero_paths_param(szp) is None

    def test_check_sampling_zero_get_and_set(self) -> None:
        """Tests the functions that set and get the sampling zeros"""
        assert self.et.sampling_zeros is None

        szp = [('Medium',), ('Medium', 'High')]
        self.et.sampling_zeros = szp
        assert self.et.sampling_zeros == szp

    def test_order_of_columns(self) -> None:
        assert self.reordered_et.variables == list(self.df.columns[::-1])

    def test_create_node_list_from_paths(self) -> None:
        paths = defaultdict(int)
        paths[('path',)] += 1
        paths[('path', 'to')] += 1
        paths[('path', 'away')] += 1
        paths[('road',)] += 1
        paths[('road', 'to')] += 1
        paths[('road', 'away')] += 1

        # code being tested:
        node_list = self.et._EventTree__create_node_list_from_paths(paths)

        print(node_list)
        assert len(list(paths.keys())) + 1 == len(node_list)
        assert node_list[0] == 's0'
        assert node_list[-1] == 's%d' % (len(node_list) - 1)

    def test_construct_event_tree(self) -> None:
        """Tests the construction of an event tree from a set of paths,
        nodes, and """
        EXPECTED_NODE_COUNT = 45
        assert len(self.et) == EXPECTED_NODE_COUNT
        assert len(self.et.edges) == EXPECTED_NODE_COUNT - 1
        edge_counts = self.et.edge_counts

        assert len(edge_counts) == EXPECTED_NODE_COUNT - 1
        for _, count in edge_counts.items():
            assert isinstance(count, int)

    def test_get_functions_producing_expected_data(self) -> None:
        edges = list(self.et.edges)
        assert isinstance(edges, list)
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 3
            assert isinstance(edge[0], str)
            assert isinstance(edge[1], str)
            assert isinstance(edge[2], str)

        check_list_contains_strings(list(self.et))
        check_list_contains_strings(self.et.situations)
        check_list_contains_strings(self.et.leaves)

        edge_counts = self.et.edge_counts
        print(edge_counts)
        assert isinstance(edge_counts, dict)
        for edge, count in edge_counts.items():
            assert isinstance(edge, tuple)
            for node in edge:
                assert isinstance(node, str)
            assert isinstance(count, int)

    def test_dataframe_with_numeric_values(self) -> None:
        """Ensures figure can be produced from dataframes with
        numeric values"""
        self.df["NewColumn"] = [1] * len(self.df)
        new_et = EventTree(self.df)
        try:
            new_et.create_figure("out/test_dataframe_with_numeric_values.pdf")
        except InvocationException:
            pass
        except Exception as err:
            raise AssertionError(
                "Could not create figure with numeric data"
            ) from err
        return None


class TestIntegration():
    def setup(self):
        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_et = EventTree(
            dataframe=self.med_df,
            sampling_zero_paths=self.med_s_z_paths
        )

        # non-stratified dataset
        fall_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/Falls_Data.xlsx')
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_et = EventTree(
            dataframe=self.fall_df,
            sampling_zero_path=self.fall_s_z_paths
        )

    def test_categories_per_variable(self) -> None:
        expected_med_cats_per_var = {
            "Classification": 2,
            "Group": 3,
            "Difficulty": 2,
            "Response": 2,
        }
        actual_med_cats_per_var = self.med_et.categories_per_variable
        assert expected_med_cats_per_var == actual_med_cats_per_var

        expected_fall_cats_per_var = {
            "HousingAssessment": 4,
            "Risk": 2,
            "Treatment": 3,
            "Fall": 2,
        }
        actual_fall_cats_per_var = self.fall_et.categories_per_variable
        assert expected_fall_cats_per_var == actual_fall_cats_per_var


def check_list_contains_strings(str_list) -> bool:
    assert isinstance(str_list, list)
    for elem in str_list:
        assert isinstance(elem, str)


class TestUsecase():
    def setup(self):
        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_et = EventTree(
            dataframe=self.med_df,
            sampling_zero_paths=self.med_s_z_paths
        )

        # non-stratified dataset
        fall_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/Falls_Data.xlsx')
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_et = EventTree(
            dataframe=self.fall_df
        )

    def test_fall_cats_per_var(self):
        expected_fall_cats_per_var = {
            "HousingAssessment": 4,
            "Risk": 2,
            "Treatment": 3,
            "Fall": 2,
        }
        actual_fall_cats_per_var = self.fall_et.categories_per_variable
        assert expected_fall_cats_per_var == actual_fall_cats_per_var


class TestChangingDataFrame():
    def setup(self):
        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_et = EventTree(
            dataframe=self.med_df,
            sampling_zero_paths=self.med_s_z_paths
        )

        # non-stratified dataset
        fall_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/Falls_Data.xlsx')
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_et = EventTree(
            dataframe=self.fall_df,
            sampling_zero_path=self.fall_s_z_paths
        )

    def test_add_empty_column(self) -> None:
        # adding empty column
        med_empty_column_df = self.med_df
        med_empty_column_df["extra"] = ""
        med_empty_column_et = EventTree(
            dataframe=med_empty_column_df
        )
        assert med_empty_column_et.adj == self.med_et.adj

        fall_empty_column_df = self.fall_df
        fall_empty_column_df["extra"] = ""
        fall_empty_column_et = EventTree(
            dataframe=fall_empty_column_df
        )
        assert fall_empty_column_et.adj == self.fall_et.adj

    def test_add_NA_column(self) -> None:
        # adding NA column
        med_add_NA_df = self.med_df
        med_add_NA_df["extra"] = np.nan
        med_add_NA_et = EventTree(
            dataframe=med_add_NA_df
        )
        assert med_add_NA_et.adj == self.med_et.adj

        fall_add_NA_df = self.fall_df
        fall_add_NA_df["extra"] = np.nan
        fall_add_NA_et = EventTree(
            dataframe=fall_add_NA_df
        )
        assert fall_add_NA_et.adj == self.fall_et.adj

    def test_add_same_column(self) -> None:
        # adding column with no more information
        med_add_same_df = self.med_df
        med_add_same_df["extra"] = "same for all"
        med_add_same_et = EventTree(
            dataframe=med_add_same_df
        )
        assert len(med_add_same_et.leaves) == len(self.med_et.leaves)

        fall_add_same_df = self.fall_df
        fall_add_same_df["extra"] = "same for all"
        fall_add_same_et = EventTree(
            dataframe=fall_add_same_df
        )
        assert len(fall_add_same_et.leaves) == len(self.fall_et.leaves)

    def test_add_same_column_int(self, mocker: MockerFixture) -> None:
        mocker.patch('pydotplus.Dot.write')
        # adding column with no more information
        med_add_same_df = self.med_df
        med_add_same_df["extra"] = 1
        med_add_same_et = EventTree(
            dataframe=med_add_same_df
        )
        med_add_same_et.create_figure("et_fig_path.pdf")
        assert len(med_add_same_et.leaves) == len(self.med_et.leaves)

        fall_add_same_df = self.fall_df
        fall_add_same_df["extra"] = 1
        fall_add_same_et = EventTree(
            dataframe=fall_add_same_df
        )
        fall_add_same_et.create_figure("et_fig_path.pdf")
        assert len(fall_add_same_et.leaves) == len(self.fall_et.leaves)
