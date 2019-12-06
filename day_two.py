import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import soloA, soloB, read_files, powerspecplot, rotate_21, which_csvs, rotate_24, shifttime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time
from datetime import datetime
import glob


def day_two(all_files, collist, soloA_bool, num, start_dt, end_dt, alt, sampling_freq = None):
    #set this to the directory where the data is kept on your local computer
    if soloA_bool:
        df = read_files(all_files, soloA_bool, jonas, sampling_freq, collist, day=2, start_dt = start_dt, end_dt = end_dt)
        rotate_mat = rotate_24(soloA_bool)[num-1]
    else:
        df = read_files(all_files, soloA_bool, jonas, sampling_freq, collist, day=2, start_dt = start_dt, end_dt = end_dt)
        rotate_mat = rotate_24(soloA_bool)[num-9]
    df.iloc[:,0:3] = np.matmul(rotate_mat, df.iloc[:,0:3].values.T).T
    print(len(df))
    
    #change time in MFSA to spacecraft time
    
    df2 = df.between_time(start_dt.time(), end_dt.time()) 
    
    print(df2.head())
    
    df2 = shifttime(df2, soloA_bool)
    
    plot = True
    #df2 = df.between_time(start_dt.time(), end_dt.time()) 
    print(df2.head())
    
    if plot: #plotting the raw probes results
        plt.figure()
        for col in collist[1:]:
            plt.plot(df2.index.time, df2[col], label=str(col))
            #print(df2[col].abs().idxmax())
        plt.xlabel('Time (s)')
        plt.ylabel('B (nT)')
        plt.title(f'Probe {num} @ {sampling_freq}Hz, {start_dt.date()}')
        plt.legend(loc="best")
        plt.show()

    fs = 50

    powerspecplot(df, fs, collist, alt)

if __name__ == "__main__":
    
    jonas = True

    if jonas:
        file_path_A = r'C:\Users\jonas\MSci-Data\powered\SoloA_2019-06-24--08-14-46_9\SoloA_2019-06-24--08-14-46_1.csv' #the first couple of files in some of the folders are from earlier days
        file_path_B = r'C:\Users\jonas\MSci-Data\powered\SoloB_2019-06-24--08-14-24_20\SoloB_2019-06-24--08-14-24_1.csv'
        path_A = r'C:\Users\jonas\MSci-Data\day_two\SoloA_2019-06-24--08-14-46_9'
        path_B = r'C:\Users\jonas\MSci-Data\day_two\SoloB_2019-06-24--08-14-24_20'
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_two\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_two\B'
    else:
        file_path_A = os.path.expanduser("~/Documents/MsciProject/Data/SoloA_2019-06-24--08-14-46_9/SoloA_2019-06-24--08-14-46_1.csv")
        file_path_B = os.path.expanduser("~/Documents/MsciProject/Data/SoloB_2019-06-24--08-14-24_20/SoloB_2019-06-24--08-14-24_1.csv")
        path_fol_A =  os.path.expanduser("~/Documents/MsciProject/Data/day_two/A")
        path_fol_B =  os.path.expanduser("~/Documents/MsciProject/Data/day_two/B")
    
    num = 12
    if num < 9:
        soloA_bool = True
    else:
        soloA_bool = False
    if num <10:
        num_str = f'0{num}'
    else: 
        num_str = num
    collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']

    start_dt = datetime(2019,6,24,9,25) + pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
    end_dt = datetime(2019,6,24,10,9) + pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)# this is the end

    day = 2
    start_csv, end_csv = which_csvs(soloA_bool, day ,start_dt, end_dt, tz_MAG=False) #this function (in processing.py) finds the number at the end of the csv files we want
    print(start_csv, end_csv)

    all_files = [0]*(end_csv + 1 - start_csv)
    

    for index, i in enumerate(range(start_csv, end_csv + 1)): #this will loop through and add the csv files that contain the start and end time set above
        if soloA_bool:
            if jonas:
                all_files[index] = path_fol_A + f'\SoloA_2019-06-24--08-14-46_{i}.csv'
            else:
                all_files[index] = path_fol_A + os.path.expanduser(f'/SoloA_2019-06-24--08-14-46_{i}.csv') #need to change path_fol_A  to the path where your A folder is
        else:
            if jonas:
                all_files[index] = path_fol_B + f'\SoloB_2019-06-24--08-14-24_{i}.csv'
            else:
                all_files[index] = path_fol_B + os.path.expanduser(f'/SoloB_2019-06-24--08-14-24_{i}.csv') #need to change path_fol_B to the path where your B folder is
    #print(all_files)
    alt = False
    day_two(all_files, collist, soloA_bool, num, start_dt, end_dt, alt, sampling_freq = 100) #pass through the list containing the file paths

