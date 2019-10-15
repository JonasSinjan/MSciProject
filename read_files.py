import pandas as pd
import numpy as np
import scipy as sp
import glob

#set this to the directory where the data is kept on your local machine
# path = r'C:\Users\jonas\MSci-Data\SoloA_2019-06-21--08-10-10_20'
# all_files = glob.glob(path + "/*.csv")

file_name = 'C:\Users\jonas\MSci-Data\SoloA_2019-06-21--08-10-10_20\SoloA_2019-06-19--16-19-29_1.csv'

df = pd.read_csv(file_name, index_col = None, header=None, error_bad_lines=False, warn_bad_lines = False)
df.head()
cols = list(df.columns.values)
print(cols)
#print(df.iloc[1]
#this concatenates all the csv files in one folder into one dataframe
#NB: for the unpowered test where it is moved through the array - unsure if should be concatenated
# li = []

# for filename in all_files:
#     df = pd.read_csv(filename, index_col = None, header=None, error_bad_lines=False, warn_bad_lines = False)
#     li.append(df)

# frame = pd.concat(li, ignore_index=True)

#print(frame.tail())