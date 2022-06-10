import os

import numpy as np
import pandas as pd

HOME = os.path.expanduser('~')
PROJ_DIR = os.getcwd()

###

def get_data(data_file: str) -> pd.DataFrame:
    data_path = os.path.join(PROJ_DIR,'data/',data_file)
    df = pd.read_csv(data_path)
    return df

def get_unique_vals(df: pd.DataFrame):
    pass