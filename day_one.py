import matplotlib.pyplot as plt
from read_merge import soloA, soloB, read_files
import pandas as pd
import os
import numpy as np
import scipy.signal as sps

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

    
    
    """
    align = False
    
    if align == True:
        
        df_A = soloA(file_path_A)
        df_B = soloB(file_path_B)
        
        collist_A = ['Probe01_X','Probe01_Y','Probe01_Z','Probe01_||','Probe02_X','Probe02_Y']
        max_index_A = []   #empty list for max abs values for each probe
        #print (df_A.iloc[0])
        for col in collist_A:
            probe = df_A[col].tolist()
            max_index_A.append(probe.index(max(probe, key = abs)))
        peak_index_A = max(max_index_A, key=max_index_A.count)   #find the mode of the list - most commonly shared max point between probes in A
        sample_rate = df_A['time'][1]-df_A['time'][0]
        peak_time_A = peak_index_A*sample_rate
        #peak_time_A = df_A['time'][peak_index_A]    #this is the time at which the largest peak occurs in A
        print("A",peak_time_A)
        
        collist_B = ['Probe10_X','Probe10_Y','Probe10_Z','Probe10_||','Probe11_X','Probe11_Y']
        max_index_B = []   #empty list for max abs values for each probe
        for col in collist_B:
            probe = df_B[col].tolist()
            max_index_B.append(probe.index(max(probe, key = abs)))
        peak_index_B = max(max_index_B, key=max_index_B.count)   #find the mode of the list - most commonly shared max point between probes in A
        peak_time_B = peak_index_B*sample_rate    #this is the time at which the largest peak occurs in B
        print("B",peak_time_B)
    """
    
    soloA_var = False

    collist = ['time', 'Probe01_X'] #'Probe01_||'
    df = read_files(path_A, soloA_var, jonas, collist)

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
            plt.plot(df['time'], df[col].tolist(), label=str(col))
            
        # for col in df.columns.tolist()[-4:0]:
        #     plt.plot(df['time'], df[col], label=str(col))

        plt.xlabel('Time (s)')
        plt.ylabel('B (nT)')
        plt.legend()
        plt.show()

    
    #power spectral density plot
    x = df['Probe01_X'][:20000]
    fs = 500 # sampling rate
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
    f, t, Sxx = sps.spectrogram(x,fs)
    plt.figure()
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectogram')
    plt.show()
    
    
day_one()





