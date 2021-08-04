from ..src.cegpy.graphs.ceg import ChainEventGraph
from ..src.cegpy.trees.staged import StagedTree
from pathlib import Path
import pandas as pd


class TestCEG(object):
    def setup(self):
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        st_params = {
            'dataframe': pd.read_excel(df_path)
        }
        st = StagedTree(st_params)
        st.calculate_AHC_transitions()
        self.ceg = ChainEventGraph(staged_tree=st)

    def test_integration(self):
        self.ceg._create_graph_representation()
