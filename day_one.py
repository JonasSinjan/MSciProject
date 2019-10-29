import matplotlib.pyplot as plt
from read_merge import soloA, soloB, read_files
from align import align
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time



def day_one(collist, soloA_bool):
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
    probe_x = collist[1]
    probe_y = collist[2]
    probe_z = collist[3]
    probe_m = collist[4]
    x = df[probe_x]#[:20000]
    f_x, Pxx_x = sps.periodogram(x,fs, scaling='spectrum')
    x = df[probe_y]#[:20000]
    f_y, Pxx_y = sps.periodogram(x,fs, scaling='spectrum')
    x = df[probe_z]#[:20000]
    f_z, Pxx_z = sps.periodogram(x,fs, scaling='spectrum')
    x = df[probe_m]#[:20000]
    f_m, Pxx_m = sps.periodogram(x,fs, scaling='spectrum')
    
    def plot_power(f,Pxx,probe):
        plt.semilogy(f,np.sqrt(Pxx)) #sqrt required for power spectrum, and semi log y axis
        plt.xlim(0,60)
        plt.ylim(10e-4,10e-1)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Log(FFT magnitude)')
        plt.title(f'{probe}')
        peaks, _ = sps.find_peaks(np.log10(np.sqrt(Pxx)), prominence = 3)
        print([round(i,1) for i in f[peaks] if i <= 20], len(peaks))
        plt.semilogy(f[peaks], np.sqrt(Pxx)[peaks], marker = 'x', markersize = 10, color='orange', linestyle = 'None')
    

    plt.figure()
    plt.title('Power Spectrum')
    plt.subplot(221)
    plot_power(f_x, Pxx_x, probe_x)
    
    plt.subplot(222)
    plot_power(f_y, Pxx_y, probe_y)

    plt.subplot(223)
    plot_power(f_z, Pxx_z, probe_z)
    
    plt.subplot(224)
    plot_power(f_m, Pxx_m, probe_m)
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    
    
    #spectogram

    x = df[collist[1]][5270000:5310000]
    fs = 1000 # sampling rate
    f, Pxx = sps.periodogram(x,fs)
    f, t, Sxx = sps.spectrogram(x,fs,nperseg=700)
    plt.figure()
    plt.pcolormesh(t, f, Sxx,vmin = 0.,vmax = 0.1)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectogram')
    fig = plt.gcf()
    plt.clim()
    plt.colorbar()  
    plt.show()
    
    
num = '02'
collist = ['time', f'Probe{num}_X', f'Probe{num}_Y', f'Probe{num}_Z', f'Probe{num}_||']
soloA_bool = True
day_one(collist, soloA_bool)





