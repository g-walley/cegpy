import __init__
from event_tree import EventTree
from pathlib import Path
import pandas as pd
import datetime

__init__.logger.info("Starting Main Test Program")

df_path = Path(__file__).resolve().parent.parent.parent.joinpath(
    'data/medical_dm_modified.xlsx')

df = pd.read_excel(df_path)
et = EventTree({'dataframe': df})

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

fig_path = Path(__file__).resolve().parent.joinpath(
    'out/medical_dm_event_tree__%s' % dt_string)
et.create_figure('')
