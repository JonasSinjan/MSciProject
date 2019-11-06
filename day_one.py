import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import soloA, soloB, read_files, powerspecplot, rotate_21, which_csvs
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from align import align
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time
from datetime import datetime
import glob


def day_one(all_files, collist, soloA_bool, num, start_dt, end_dt):
    #set this to the directory where the data is kept on your local computer
  
    if soloA_bool:
        df = read_files(all_files, soloA_bool, jonas, collist, day=1, start_dt = start_dt, end_dt = end_dt)
        rotate_mat = rotate_21(soloA_bool)[num-1]
    else:
        df = read_files(all_files, soloA_bool, jonas, collist, day=1, start_dt = start_dt, end_dt = end_dt)
        rotate_mat = rotate_21(soloA_bool)[num-9]
    df.iloc[:,0:3] = np.matmul(rotate_mat, df.iloc[:,0:3].values.T).T
    print(len(df))
    
    #time_diff = align(file_path_A, file_path_B)
    #print(time_diff)
    #now need to use pd.timedelta to subtract/add this time to the datetime object column 'time' in the df
    
    plot = True

    if plot: #plotting the raw probes results
        df2 = df.between_time(start_dt.time(), end_dt.time())
        plt.figure()
        for col in collist[1:]:
            plt.plot(df2.index.to_pydatetime(), df2[col], label=str(col))
            print(df2[col].idxmax())
        plt.xlabel('Time (s)')
        plt.ylabel('B (nT)')
        plt.title(f'Probe {num}, {start_dt.date()}')
        plt.legend()
        plt.show()

    #fs = 100
    #powerspecplot(df, fs, collist)
    
    #spectogram
    """
    x = df[collist[1]][5270000:5310000]
    #fs = 200 # sampling rate
    #f, Pxx = sps.periodogram(x,fs)
    f, t, Sxx = sps.spectrogram(x,fs)#,nperseg=700)
    plt.figure()
    plt.pcolormesh(t, f, Sxx,vmin = 0.,vmax = 0.1)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectogram')
    plt.clim()
    fig = plt.gcf()
    plt.colorbar()  
    plt.show()
    
# num = '07'
# collist = ['time', f'Probe{num}_X', f'Probe{num}_Y', f'Probe{num}_Z', f'Probe{num}_||']
# soloA_bool = True
# day_one(collist, soloA_bool)
"""

if __name__ == "__main__":

    jonas = True

    if jonas:
        file_path_A = r'C:\Users\jonas\MSci-Data\day_one\A\SoloA_2019-06-21--08-10-10_20\SoloA_2019-06-21--08-10-10_1.csv'
        file_path_B = r'C:\Users\jonas\MSci-Data\day_one\B\SoloB_2019-06-21--08-09-10_20\SoloB_2019-06-21--08-09-10_1.csv'
        path_A = r'C:\Users\jonas\MSci-Data\day_one\A\SoloA_2019-06-21--08-10-10_50'
        path_B = r'C:\Users\jonas\MSci-Data\day_one\B\SoloB_2019-06-21--08-09-10_20'
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_one\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_one\B'
    else:
        file_path_A = os.path.expanduser("~/Documents/MSciProject/Data/SoloA_2019-06-21--08-10-10_20/SoloA_2019-06-21--08-10-10_01.csv")
        file_path_B = os.path.expanduser("~/Documents/MSciProject/Data/SoloB_2019-06-21--08-09-10_20/SoloB_2019-06-21--08-09-10_01.csv")
        path_A = os.path.expanduser("~/Documents/MSciProject/Data/SoloA_2019-06-21--08-10-10_50")
        path_B = os.path.expanduser("~/Documents/MSciProject/Data/SoloB_2019-06-21--08-09-10_20")

    num = 4
    soloA_bool = True
    if num <10:
        num_str = f'0{num}'
    else: 
        num_str = num
    collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']

    start_dt = datetime(2019,6,21,10,58)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
    end_dt = datetime(2019,6,21,10,58,50)# this is the end
    day = 1
    start_csv, end_csv = which_csvs(soloA_bool, day ,start_dt, end_dt) #this function (in processing.py) finds the number at the end of the csv files we want
    print(start_csv, end_csv)

    all_files = [0]*(end_csv + 1 -start_csv)

    for index, i in enumerate(range(start_csv, end_csv + 1)): #this will loop through and add the csv files that contain the start and end time set above
        if soloA_bool:
            if jonas:
                all_files[index] = path_fol_A + f'\SoloA_2019-06-21--08-10-10_{i}.csv'
            else:
                all_files[index] = path_fol_A + os.path.expanduser(f'/SoloA_2019-06-21--08-10-10_{i}.csv') #need to change path_fol_A  to the path where your A folder is
        else:
            if jonas:
                all_files[index] = path_fol_B + f'\SoloB_2019-06-21--08-09-10_{i}.csv'
            else:
                all_files[index] = path_fol_B + os.path.expanduser(f'/SoloB_2019-06-21--08-09-10_{i}.csv') #need to change path_fol_B to the path where your B folder is
    #print(all_files)
    day_one(all_files, collist, soloA_bool, num, start_dt, end_dt) #pass through the list containing the file paths




