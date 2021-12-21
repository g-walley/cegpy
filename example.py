import logging
from src.cegpy.trees.staged import StagedTree
from src.cegpy.trees.event import EventTree
from src.cegpy import logger
from pathlib import Path
import pandas as pd
from datetime import datetime


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


# FALLS
df_path = Path(__file__).resolve().parent.joinpath(
    'data/Falls_Data.xlsx')
logger.info(str(df_path))
df = pd.read_excel(df_path)
falls_et = EventTree(dataframe=df)
falls_st = StagedTree(dataframe=df)
colours = [
    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
    '#80b1d3', '#fdb462', '#b3de69', '#fccde5'
]
et_fig_path = create_path('out/falls_event_tree', True, 'pdf')
st_fig_path = create_path('out/falls_staged_tree', True, 'pdf')
falls_et.create_figure(et_fig_path)
falls_st.calculate_AHC_transitions(colour_list=colours)
falls_st.create_figure(st_fig_path)

# get the dot graph
falls_st_dot = falls_st.dot_graph
# modify nodes and edges
falls_st_dot.get_edge("s1", "s5")[0].set_style("dotted")
falls_st_dot.get_edge("s1", "s6")[0].set_style("dotted")
falls_st_dot.get_node("s1")[0].set_shape("square")
# save as a pdf
falls_st_dot.write("out/falls_modified_staged_tree.pdf", format="pdf")
# or save as a dot
falls_st_dot.write("out/falls_modified_staged_tree.dot", format="dot")
