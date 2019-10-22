import pandas as pd
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
import os

def read_files(path, soloA, jonas, collist=None):
    #path - location of folder to concat
    #soloA - set to True if soloA, if soloB False 
    if jonas: 
        all_files = glob.glob(path + "\*.csv")
    else: 
        all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:   

        df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';', usecols = collist, nrows = 10)
        
        if collist == None:
            cols = df.columns.tolist()
            if soloA:
                new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #this will reorder the columns into the correct order
            else:
                new_cols = cols[9:13] + cols[1:9] + cols[13:17]
            df = df[new_cols]
        #print(df['time'].iloc[0])
        li.append(df)
        
    df = pd.concat(li, ignore_index = True)
    df = df.sort_values('time', ascending = True, kind = 'mergesort')
    
    return df
    
def soloA(file_path):
    #skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
    df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';')
    cols = df.columns.tolist()
    new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #reorder the columns into the correct order
    df = df[new_cols]
    return df

def soloB(file_path):
    #skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
    df_B = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';')
    cols = df_B.columns.tolist()
    new_cols = cols[9:13] + cols[1:9] + cols[13:17]#reorder the columns into the correct order
    df_B = df_B[new_cols]
    return df_B