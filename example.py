import logging
from src.cegpy.trees.staged import StagedTree
from src.cegpy.trees.event import EventTree
from src.cegpy import logger
from pathlib import Path
import pandas as pd
from datetime import datetime
import pickle
# from time import sleep


def create_path(filename, add_time=False, filetype='png'):
    dt_string = ''
    if add_time:
        now = datetime.now()
        dt_string = now.strftime("__%d-%m-%Y_%H-%M-%S")

    fig_path = Path(__file__).resolve().parent.joinpath(
            filename + dt_string + '.' + filetype)
    return fig_path


# MED
logger.setLevel(level=logging.WARNING)
logger.info("Starting Main Test Program")
df_path = Path(__file__).resolve().parent.joinpath(
    'data/medical_dm_modified.xlsx')
logger.info(str(df_path))
df = pd.read_excel(df_path)
med_et = EventTree(dataframe=df)
med_st = StagedTree(dataframe=df)

et_fig_path = create_path('out/medical_dm_event_tree', True, 'pdf')
st_fig_path = create_path('out/medical_dm_staged_tree', True, 'pdf')
med_et.create_figure(et_fig_path)
med_st.calculate_AHC_transitions()
med_st.create_figure(st_fig_path)

# pickle_path = create_path('out/med_st', True, 'pickle')
# with open(pickle_path, 'wb') as f:
#     pickle.dump(med_st, f, pickle.HIGHEST_PROTOCOL)

# with open(pickle_path, 'rb') as f:
#     med_st2 = pickle.load(f)
#     med_st2.create_figure(st_fig_path)


# # FALLS
df_path = Path(__file__).resolve().parent.joinpath(
    'data/Falls_Data.xlsx')
logger.info(str(df_path))
df = pd.read_excel(df_path)
falls_et = EventTree(dataframe=df)
falls_st = StagedTree(dataframe=df)

et_fig_path = create_path('out/falls_event_tree', True, 'pdf')
st_fig_path = create_path('out/falls_staged_tree', True, 'pdf')
falls_et.create_figure(et_fig_path)
falls_st.calculate_AHC_transitions()
falls_st.create_figure(st_fig_path)
