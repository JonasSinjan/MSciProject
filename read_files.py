import pandas as pd
import numpy as np
import scipy as sp
import glob

#set this to the directory where the data is kept on your local machine
path = r'C:\Users\jonas\MSci-Data\SoloA_2019-06-21--08-10-10_20'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col = None, header=None, error_bad_lines=False, warn_bad_lines = False)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

