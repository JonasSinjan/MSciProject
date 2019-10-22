import pandas as pd
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
import os

def read_files(path, soloA):
    #path - location of folder to concat
    #soloA - set to True if soloA, if soloB False

    #these files will be too big and take too long - should specify which columns we desire first, to only read in data we need to analysis
    all_files = sorted(glob.glob(path + "\*.csv"), key=os.path.getmtime)
    print(all_files)
    li, length_var = [], 0
    print(li)
    #while len(li) < 20:
    for index, filename in enumerate(all_files):   
        df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';', nrows = 1, )
        #print(df)
        #print(index)
        cols = df.columns.tolist()
        if soloA:
            new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #this will reorder the columns into the correct order
        else:
            new_cols = cols[9:13] + cols[1:9] + cols[13:17]
        df = df[new_cols]
        print(df['time'].iloc[0])
        li.append(df)
        
    df = pd.concat(li, ignore_index = True)
    #df.set_index('time')
    #df.sort_values(by = ['time'], ascending = True, kind = 'mergesort')
    
    
    
        
        #     #print(df['time'].iloc[0])
        #     if df['time'].iloc[0] == 0.00 and len(li) < 1:
        #         #print('check1')
        #         if soloA:
        #             new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #this will reorder the columns into the correct order
        #         else:
        #             new_cols = cols[9:13] + cols[1:9] + cols[13:17]
        #         df = df[new_cols]
        #         #print(df['time'].iloc[0])
        #         #print('check end')
        #         li.append(df)
                
        #     else:        
        #         print(index)
        #         if df['time'].iloc[0] < df_check['time'].iloc[0] + 500:
        #             if soloA:
        #                 new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #this will reorder the columns into the correct order
        #             else:
        #                 new_cols = cols[9:13] + cols[1:9] + cols[13:17]
        #             df = df[new_cols]
        #             #print(df['time'].iloc[0])
        #             li.append(df)
        #         else:
        #             continue
            
        #     print(df['time'].iloc[0])            
        #     li.append(df)
        #     df_check = df
        #     length_var += len(df)
            
        
        # df_check = li[-1]
            
        #print(df['time'].iloc[0:round(len(df)/30)].mean())
        #print('og = ', len(df))
        #df = df.groupby(np.arange(len(df))//30).mean()
        #print(df['time'].iloc[0])
        #print('compressed = ', len(df))
        #cols = df.columns.tolist()
        #if soloA:
        #    new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #this will reorder the columns into the correct order
        #else:
        #    new_cols = cols[9:13] + cols[1:9] + cols[13:17]
        #df = df[new_cols]
        #print(df.head())
        
        #print('total len = ', length_var)

    # df = pd.concat(li, ignore_index=True)
    # df.sort_values()
    #print(1/df['time'].iloc[1]) #the new effective sampling rate
    
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