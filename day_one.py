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
import csv


def day_one(all_files, collist, soloA_bool, num, start_dt, end_dt, alt, sampling_freq = None):
    """
    all_files - set this to the directory where the data is kept on your local computer
    collist - list of columns desired
    soloA_bool - set to True if desire sensors in SoloA channel, else False
    num - set to the number of the probe desired
    start_dt - start datetime desired
    end_dt - end datetime desired
    alt - boolean - set to True if desire the 'brute force' method for power spectrum rather than periodogram method
    sampling_freq - set to desired sampling frequency - default = None
    """
    if soloA_bool:
        df = processing.read_files(all_files, soloA_bool, windows, sampling_freq, collist, day=1, start_dt = start_dt, end_dt = end_dt)
        rotate_mat = processing.rotate_21(soloA_bool)[num-1]
    else:
        df = processing.read_files(all_files, soloA_bool, windows, sampling_freq, collist, day=1, start_dt = start_dt, end_dt = end_dt)
        rotate_mat = processing.rotate_21(soloA_bool)[num-9]
    df.iloc[:,0:3] = np.matmul(rotate_mat, df.iloc[:,0:3].values.T).T
    print(len(df))
    
    plot = True
    df2 = df.between_time(start_dt.time(), end_dt.time())
    df3 = df2.resample('1s').mean()
    #print(df2.head())

    if plot: #plotting the raw probes results
        #plt.figure()
        tmp = []
        for col in collist[1:]:
            plt.plot(df2.index.time, df2[col], label=str(col))
            #print(df2[col].abs().idxmax())
            var_1hz = np.std(df3[col])
            var_1khz = np.std(df2[col])
            print('std - 1Hz', col, var_1hz)
            print('std - 1kHz', col,  var_1khz)
            tmp.append(var_1hz)
            tmp.append(var_1khz)
        """
        plt.xlabel('Time (s)')
        plt.ylabel('B (nT)')
        plt.title(f'Probe {num} @ {sampling_freq}Hz, {start_dt.date()}')
        plt.legend(loc="best")
        plt.show()
        """
        plt.show()
    #power spectrum
    fs = sampling_freq
    #processing.powerspecplot(df, fs, collist, alt)

    """
    #spectogram    
    x = df2[collist[1]]
    fs = sampling_freq
    #f, Pxx = sps.periodogram(x,fs)
    f, t, Sxx = sps.spectrogram(x,fs)#,nperseg=700)
    plt.figure()
    plt.pcolormesh(t, f, Sxx,vmin = 0.,vmax = 0.1)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram: Probe {num} @ {sampling_freq}Hz, {start_dt.date()}')
    plt.clim()
    fig = plt.gcf()
    plt.colorbar()  
    plt.show()
    """
    return tmp


if __name__ == "__main__":

    windows = False

    if windows:
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_one\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_one\B'
    else:
        path_fol_A = os.path.expanduser("~/Documents/MSciProject/Data/day_one/A")
        path_fol_B = os.path.expanduser("~/Documents/MSciProject/Data/day_one/B")

    alt = False #set to true if you want to see power spec using the stnadard method - not the inbuilt funciton
    #num = 5
    b_noise = []
    for num in [1]:#range(1,13):
        print('num = ', num)
        if num < 9:
            soloA_bool = True
        else:
            soloA_bool = False
        if num < 10:
            num_str = f'0{num}'
        else: 
            num_str = num
        collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']

        start_dt = datetime(2019,6,21,14,30) + pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 30, milliseconds = 137)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
        end_dt = datetime(2019,6,21,15,30) + pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 30, milliseconds = 137)# this is the end

        day = 1
        start_csv, end_csv = processing.which_csvs(soloA_bool, day ,start_dt, end_dt) #this function (in processing.py) finds the number at the end of the csv files we want
        print(start_csv, end_csv)

        all_files = [0]*(end_csv + 1 - start_csv)

        for index, i in enumerate(range(start_csv, end_csv + 1)): #this will loop through and add the csv files that contain the start and end time set above

            if soloA_bool:
                if windows:
                    all_files[index] = path_fol_A + f'\SoloA_2019-06-21--08-10-10_{i}.csv'
                else:
                    all_files[index] = path_fol_A + os.path.expanduser(f'/SoloA_2019-06-21--08-10-10_{i}.csv') #need to change path_fol_A  to the path where your A folder is
            else:
                if windows:
                    all_files[index] = path_fol_B + f'\SoloB_2019-06-21--08-09-10_{i}.csv'
                else:
                    all_files[index] = path_fol_B + os.path.expanduser(f'/SoloB_2019-06-21--08-09-10_{i}.csv') #need to change path_fol_B to the path where your B folder is

        tmp = day_one(all_files, collist, soloA_bool, num, start_dt, end_dt, alt, sampling_freq = 1000) #pass through the list containing the file paths
        b_noise.extend(tmp)

    """
    w = csv.writer(open(f"day1_mfsa_probe_vars.csv", "w"))
    w.writerow(["Probe","Bx_var","By_var","Bz_var","Bx_var_1k","By_var_1k","Bz_var_1k"])
    val = b_noise
    j = 0
    for i in range(12):
        w.writerow([i+1,val[j],val[j+2],val[j+4],val[j+1],val[j+3],val[j+5]])#,val[9],val[10],val[11]])
        j += 6
    """
    

