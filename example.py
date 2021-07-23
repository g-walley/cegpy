import logging
from src.cegpy.trees.event import EventTree
from src.cegpy import logger
from pathlib import Path
import pandas as pd
from datetime import datetime


logger.setLevel(level=logging.DEBUG)
logger.info("Starting Main Test Program")
df_path = Path(__file__).resolve().parent.joinpath(
    'data/medical_dm_modified.xlsx')
logger.info(str(df_path))
df = pd.read_excel(df_path)
et = EventTree({'dataframe': df})

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
fig_path = Path(__file__).resolve().parent.joinpath(
    'out/medical_dm_event_tree__%s' % dt_string)
et.create_figure("image")
et.create_figure(fig_path)
