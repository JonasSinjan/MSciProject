import pandas as pd
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt

def soloA(file_path):
    #skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
    df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';')
    #print(df.head())
    cols = df.columns.tolist()
    #this will reorder the columns into the correct order
    new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9]
    df = df[new_cols]

    return df

def soloB(file_path):
    #skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
    df_B = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';')

    cols = df_B.columns.tolist()
    #reorder the columns into the correct order
    new_cols = cols[9:13] + cols[1:9] + cols[13:17]
    df_B = df_B[new_cols]

    return df_B

def concatenate(a,b):
    df = pd.concat([a, b], axis = 1)
    #print(len(df))
    #trying to average over every 50 rows, but unsure how to group and cant get it to work
    # if df['time'].iloc[1] == 0.001:
    #     #df = df.groupby(np.arange(round(len(df), -3))//50).mean()
    #     df = df.groupby(df.index // 50).mean()
    #     #df = df.groupby(50).mean()

    #print(df.head())
    return df

    # collist = df.columns.tolist()
    # df_new = pd.DataFrame(np.nan, index=range(len(df)), columns=collist)
    
    # index = 0
    
    # for index, row in df.iterrows()

    #     index += 1
    #the code below was an attempt at combining all the csv files into one dataframe - but this isn't working yet so you can ignore it

    # path = r'C:\Users\jonas\MSci-Data\SoloA_2019-06-21--08-10-10_20'
    # all_files = glob.glob(path + "/*.csv")
    #print(df.iloc[1]
    #this concatenates all the csv files in one folder into one dataframe
    #NB: for the unpowered test where it is moved through the array - unsure if should be concatenated
    # li = []
    # for filename in all_files:
    #     df = pd.read_csv(filename, index_col = None, header=None, error_bad_lines=False, warn_bad_lines = False)
    #     li.append(df)
    # frame = pd.concat(li, ignore_index=True)
    #print(frame.tail())
