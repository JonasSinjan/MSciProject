import matplotlib.pyplot as plt
import matplotlib as mpl
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
        path_A = os.path.expanduser("~/Documents/MSciProject/Data/SoloA_2019-06-21--08-10-10_50")
        path_B = os.path.expanduser("~/Documents/MSciProject/Data/SoloB_2019-06-21--08-09-10_20")

    if soloA_bool:
        df = read_files(path_A, soloA_bool, jonas, collist)
    else:
        df = read_files(path_B, soloA_bool, jonas, collist)
    print(len(df))
    
    time_diff = align(file_path_A, file_path_B)
    print(time_diff)
    #now need to use pd.timedelta to subtract/add this time to the datetime object column 'time' in the df
    
    plot = True

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
    fs = 100 # sampling rate
    probe_x = collist[1]
    probe_y = collist[2]
    probe_z = collist[3]
    probe_m = collist[4]
    x = df[probe_x][:20000]
    f_x, Pxx_x = sps.periodogram(x,fs, scaling='spectrum')
    x_y = df[probe_y][:20000]
    f_y, Pxx_y = sps.periodogram(x_y,fs, scaling='spectrum')
    x_z = df[probe_z][:20000]
    f_z, Pxx_z = sps.periodogram(x_z,fs, scaling='spectrum')
    x_m = df[probe_m][:20000]
    f_m, Pxx_m = sps.periodogram(x_m,fs, scaling='spectrum')
    x_t = x + x_y + x_z
    f_t, Pxx_t = sps.periodogram(x_t, fs, scaling = 'spectrum')
    
    def plot_power(f,Pxx,probe):
        plt.semilogy(f,np.sqrt(Pxx)) #sqrt required for power spectrum, and semi log y axis
        plt.xlim(0,40)
        plt.ylim(10e-5,10e-1)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Log(FFT magnitude)')
        plt.title(f'{probe}')
        peaks, _ = sps.find_peaks(np.log10(np.sqrt(Pxx)), prominence = 2)
        print([round(i,1) for i in f[peaks] if i <= 20], len(peaks))
        plt.semilogy(f[peaks], np.sqrt(Pxx)[peaks], marker = 'x', markersize = 10, color='orange', linestyle = 'None')
    

    plt.figure()
    mpl.rcParams['agg.path.chunksize'] = 10000
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
    
    plt.figure()
    Trace = 'Trace'
    plot_power(f_t, Pxx_t, Trace)
    
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
    
    

num = '07'
collist = ['time', f'Probe{num}_X', f'Probe{num}_Y', f'Probe{num}_Z', f'Probe{num}_||']
soloA_bool = True
day_one(collist, soloA_bool)

"""
if __name__ == "__main__":
    num = '12'
    soloA_bool = False
    collist = ['time', f'Probe{num}_X', f'Probe{num}_Y', f'Probe{num}_Z', f'Probe{num}_||']
    day_one(collist, soloA_bool)
"""





