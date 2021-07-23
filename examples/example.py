import cegpy
import logging
from cegpy.trees.event import EventTree
from pathlib import Path
import pandas as pd
from datetime import datetime

cegpy.logger.setLevel(level=logging.DEBUG)
cegpy.logger.info("Starting Main Test Program")


df_path = Path(__file__).resolve().parent.parent.joinpath(
    'data/medical_dm_modified.xlsx')

df = pd.read_excel(df_path)
et = EventTree({'dataframe': df})

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

fig_path = Path(__file__).resolve().parent.joinpath(
    'out/medical_dm_event_tree__%s' % dt_string)

cegpy.logger.info("parent: %s" % str(fig_path))
et.create_figure(fig_path)
