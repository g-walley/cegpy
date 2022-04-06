import pandas as pd
from cegpy import StagedTree

data = pd.read_csv("data/Asym.csv")
data = data.astype(str)
missing_paths = [
    ('0', '1', '1', '1'),
    ('1', '0', '1', '1'),
    ('0', '1', '0', '1')
]
st = StagedTree(data, sampling_zero_paths=missing_paths)
