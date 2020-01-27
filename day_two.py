import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import processing
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
        df = processing.read_files(all_files, soloA_bool, windows, sampling_freq, collist, day=2, start_dt = start_dt, end_dt = end_dt)
        rotate_mat = processing.rotate_24(soloA_bool)[num-1]
    else:
        df = processing.read_files(all_files, soloA_bool, windows, sampling_freq, collist, day=2, start_dt = start_dt, end_dt = end_dt)
        rotate_mat = processing.rotate_24(soloA_bool)[num-9]
    df.iloc[:,0:3] = np.matmul(rotate_mat, df.iloc[:,0:3].values.T).T
    print(len(df))
    
    #find the df of the exact time span desired
    df2 = df.between_time(start_dt.time(), end_dt.time()) 
    
    print(df2.head())
    
    #shitfing time so the axes are in spacecraft time to compare with current data
    df2 = processing.shifttime(df2, soloA_bool)
    
    plot = True
    #df2 = df.between_time(start_dt.time(), end_dt.time()) 
    print(df2.head())
    
    if plot: #plotting the raw probes results
        plt.figure()
        #for col in collist[2:4:3]:
        plt.plot(df2.index.time, df2[collist[1]]+6, label=str(collist[1])) #probe 10: +6 #probe 9: +203
        plt.plot(df2.index.time, df2[collist[2]]-47, label=str(collist[2])) #probe 10: -47 #probe 9: -50
        plt.plot(df2.index.time, df2[collist[3]]+64, label=str(collist[3])) #probe 10: +64 #probe 9: +104
            #print(df2[col].abs().idxmax())
        plt.xlabel('Spacecraft Time [H:M:S]')
        plt.ylabel('dB [nT]')
        plt.title(f'Probe {num} @ {sampling_freq}Hz, {start_dt.date()} PHI')
        plt.legend(loc="best")
        plt.show()

    fs = 50

    processing.powerspecplot(df, fs, collist, alt)

if __name__ == "__main__":
    
    windows = True

    if windows:
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_two\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_two\B'
    else:
        path_fol_A =  os.path.expanduser("~/Documents/MsciProject/Data/day_two/A")
        path_fol_B =  os.path.expanduser("~/Documents/MsciProject/Data/day_two/B")
    
    #here select which probe is desired, only one at a time
    num = 10
    if num < 9:
        soloA_bool = True
    else:
        soloA_bool = False
    if num < 10:
        num_str = f'0{num}'
    else: 
        num_str = num
    collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']
    #8:03 - 8:40 phi, 9:24-10:10 eui
    #the datetime we change here is in spacecraft time - used for if want probes for a certain current profile (which is in spacecraft time)
    start_dt = datetime(2019,6,24,8,3) + pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
    end_dt = datetime(2019,6,24,8,40) + pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)# this is the end
    #start and end dt now in MFSA (German UT) time - as MFSA in that time
    day = 2
    #finding the correct MFSA data files
    start_csv, end_csv = processing.which_csvs(soloA_bool, day ,start_dt, end_dt, tz_MAG=False) #this function (in processing.py) finds the number at the end of the csv files we want
    print(start_csv, end_csv)

    all_files = [0]*(end_csv + 1 - start_csv)
    
    for index, i in enumerate(range(start_csv, end_csv + 1)): #this will loop through and add the csv files that contain the start and end time set above
        if soloA_bool:
            if windows:
                all_files[index] = path_fol_A + f'\SoloA_2019-06-24--08-14-46_{i}.csv'
            else:
                all_files[index] = path_fol_A + os.path.expanduser(f'/SoloA_2019-06-24--08-14-46_{i}.csv') #need to change path_fol_A  to the path where your A folder is
        else:
            if windows:
                all_files[index] = path_fol_B + f'\SoloB_2019-06-24--08-14-24_{i}.csv'
            else:
                all_files[index] = path_fol_B + os.path.expanduser(f'/SoloB_2019-06-24--08-14-24_{i}.csv') #need to change path_fol_B to the path where your B folder is
    
    alt = False #if want powerspec from `brute force' method - or inbuilt scipy periodogram method
    day_two(all_files, collist, soloA_bool, num, start_dt, end_dt, alt, sampling_freq = 0.1) #pass through the list containing the file paths

