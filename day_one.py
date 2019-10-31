import matplotlib.pyplot as plt
from read_merge import soloA, soloB, read_files, powerspecplot
from align import align
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time


def day_one(collist, soloA_bool):
    #set this to the directory where the data is kept on your local computer
    jonas = True

    if jonas:
        file_path_A = r'C:\Users\jonas\MSci-Data\day_one\SoloA_2019-06-21--08-10-10_20\SoloA_2019-06-21--08-10-10_1.csv'
        file_path_B = r'C:\Users\jonas\MSci-Data\day_one\SoloB_2019-06-21--08-09-10_20\SoloB_2019-06-21--08-09-10_1.csv'
        path_A = r'C:\Users\jonas\MSci-Data\day_one\SoloA_2019-06-21--08-10-10_20'
        path_B = r'C:\Users\jonas\MSci-Data\day_one\SoloB_2019-06-21--08-09-10_20'
    else:
        file_path_A = os.path.expanduser("~/Documents/MSciProject/Data/SoloA_2019-06-21--08-10-10_20/SoloA_2019-06-21--08-10-10_01.csv")
        file_path_B = os.path.expanduser("~/Documents/MSciProject/Data/SoloB_2019-06-21--08-09-10_20/SoloB_2019-06-21--08-09-10_01.csv")
        path_A = os.path.expanduser("~/Documents/MSciProject/Data/SoloA_2019-06-21--08-10-10_20")
        path_B = os.path.expanduser("~/Documents/MSciProject/Data/SoloB_2019-06-21--08-09-10_20")

    if soloA_bool:
        df = read_files(path_A, soloA_bool, jonas, collist)
    else:
        df = read_files(path_B, soloA_bool, jonas, collist)
    print(len(df))
    
    time_diff = align(file_path_A, file_path_B)
    print(time_diff)
    
    plot = False

    if plot:
        #plotting the raw probes results
        plt.figure()
        for col in collist[1:]:
            plt.plot(df['time'], df[col], label=str(col))
            
        # for col in df.columns.tolist()[-4:0]:
        #     plt.plot(df['time'], df[col], label=str(col))
        plt.xlabel('Time (s)')
        plt.ylabel('B (nT)')
        plt.legend()
        plt.show()

    
    #power spectral density plot
    fs = 500 # sampling rate
    powerspecplot(df, fs, collist)
    
    #spectogram

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
    
if __name__ == "__main__":
    num = '02'
    collist = ['time', f'Probe{num}_X', f'Probe{num}_Y', f'Probe{num}_Z', f'Probe{num}_||']
    soloA_bool = True #if 9 or above must set to False
    day_one(collist, soloA_bool)





