import glob
import pandas as pd
import numpy as np

def soloB_concat(path):
    #file_path = r'C:\Users\jonas\MSci-Data\SoloA_2019-06-21--08-10-10_20'
    all_files = glob.glob(path + "/*.csv")
    li = []
    i = 0
    for filename in all_files:
        if i == 0:
            df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';')
            df = df.groupby(np.arange(len(df))//50).mean()
            print(df['time'].iloc[1])
        else:
            df = pd.read_csv(filename, header = None, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';')
            df = df.groupby(np.arange(len(df))//50).mean()
            print(df['time'].iloc[1])
        cols = df_B.columns.tolist()
        #reorder the columns into the correct order
        new_cols = cols[9:13] + cols[1:9] + cols[13:17]
        df_B = df_B[new_cols]
        li.append(df)
        i += 1
    df_A = pd.concat(li, axis=1, ignore_index=True)
    
    print(df_A.columns.tolist())
    
    return df_A