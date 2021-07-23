from ceg_util import CegUtil as util
import pandas as pd


def param_test(params):
    dataframe = params.get("dataframe")

    try:
        variables = list(dataframe.columns)
        print(variables)
    except AttributeError:
        raise ValueError("Required Parameter: No Dataframe provided. ")


if __name__ == "__main__":
    df_path = "%s/data/medical_dm_modified.xlsx" % util.get_package_root()
    df = pd.read_excel(df_path)

    param_test({'dataframe': df})
    param_test({'dataframes': df})
