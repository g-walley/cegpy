import logging
from src.cegpy.trees.staged import StagedTree
from src.cegpy.trees.event import EventTree
from src.cegpy import logger
from pathlib import Path
import pandas as pd
from datetime import datetime
# from time import sleep

# MED
logger.setLevel(level=logging.DEBUG)
logger.info("Starting Main Test Program")
df_path = Path(__file__).resolve().parent.joinpath(
    'data/medical_dm_modified.xlsx')
logger.info(str(df_path))
df = pd.read_excel(df_path)
med_et = EventTree({'dataframe': df})
med_st = StagedTree({'dataframe': df})

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
et_fig_path = Path(__file__).resolve().parent.joinpath(
    'out/medical_dm_event_tree__%s' % dt_string)
st_fig_path = Path(__file__).resolve().parent.joinpath(
    'out/medical_dm_staged_tree__%s' % dt_string)
med_et.create_figure(et_fig_path)
med_st.calculate_AHC_transitions()
med_st.create_figure(st_fig_path)


# FALLS
df_path = Path(__file__).resolve().parent.joinpath(
    'data/Falls_Data.xlsx')
logger.info(str(df_path))
df = pd.read_excel(df_path)
falls_et = EventTree({'dataframe': df})
falls_st = StagedTree({'dataframe': df})
# st = StagedTree({'dataframe': df})
# st1 = StagedTree({'dataframe': df, 'alpha': 4})

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
et_fig_path = Path(__file__).resolve().parent.joinpath(
    'out/falls_event_tree__%s' % dt_string)
st_fig_path = Path(__file__).resolve().parent.joinpath(
    'out/falls_staged_tree__%s' % dt_string)
falls_et.create_figure(et_fig_path)
falls_st.calculate_AHC_transitions()
falls_st.create_figure(st_fig_path)

# fig_path = Path(__file__).resolve().parent.joinpath(
#     'out/falls_staged_tree__%s' % dt_string)
# st.


# Un-comment this when using Docker so that you can look
# at the output files
# while True:
#     sleep(1)
#     pass
