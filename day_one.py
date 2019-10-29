import matplotlib.pyplot as plt
from read_merge import soloA, soloB, read_files
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time

def day_one():
    #set this to the directory where the data is kept on your local computer
    jonas = False

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


    soloA_var = True


    collist = ['time', 'Probe01_X'] #'Probe01_||'
    df = read_files(path_A, soloA_var, jonas, collist)
    print(len(df))
    
    """
    #now read in all soloB files
    collist = ['time', 'Probe10_X'] #ALWAYS READ IN TIME
    soloA_var = False
    df_2 = read_files(path_B, soloA_var, jonas, collist)
    print(len(df_2))
    """
    
    #print(df)
    #print(len(df))
    
    
    #df_A = soloA(file_path_A)
    #df_B = soloB(file_path_B)

    #df = concatenate(df_A, df_B)
    #print(df.head())

    #print(df[df['time']==1.00].index) #returns index 20 - proves that this data file is already sampled at 20Hz.


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
    x = df['Probe01_X'][1070000:1100000]
    fs = 1000 # sampling rate
    f, Pxx = sps.periodogram(x,fs)
    plt.figure()
    plt.semilogy(f,np.sqrt(Pxx)) #sqrt required for power spectrum, and semi log y axis
    #plt.xlim(0,100)
    plt.ylim(10e-2,10e1)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Linear spectrum')
    plt.title('Power Spectrum')
    plt.show()
    
    #spectogram
    f, t, Sxx = sps.spectrogram(x,fs,nperseg=1000)
    plt.figure()
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectogram')
    fig = plt.gcf()
    plt.clim()
    plt.colorbar()
    plt.show()
    
    
day_one()





