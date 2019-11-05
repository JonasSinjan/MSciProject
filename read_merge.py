import pandas as pd
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
import os
import scipy.signal as sps
from datetime import datetime, timedelta
import time

def read_files(path, soloA, jonas, collist=None):
    #path - location of folder to concat
    #soloA - set to True if soloA, if soloB False 
    if jonas: 
        all_files = glob.glob(path + "\*.csv")
    else: 
        all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        #time = pd.read_csv(filename, skiprows = 7, nrows = 1, header = None)
        #start_time = filename.strip('-')        
        if soloA:
            if collist == None:
                df =  pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';')
                cols = df.columns.tolist()
                new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #this will reorder the columns into the correct order
                df = df[new_cols]
            else:
                df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';', usecols = collist)
        else:
            if collist == None:
                df =  pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';')
                cols = df.columns.tolist()
                new_cols = [cols[0]] + cols[9:13] + cols[1:9] + cols[13:17]
                df = df[new_cols]
            else:
                df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';', usecols = collist)
            
        li.append(df)
        
    df = pd.concat(li, ignore_index = True, sort=True)

    
    start = time.process_time()
    if soloA:
        if '21' in all_files[0]: #for day_one
            df['time'] = df['time'] + 10.12
            df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-21 08:10:00' )
        elif '24' in all_files[0]: #for day_two
            df['time'] = df['time'] + 46.93
            df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-24 08:14:00' )
    else:
        if '21' in all_files[0]:
            df['time'] = df['time'] + 10
            df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-21 08:09:00' )
        elif '24' in all_files[0]:
            df['time'] = df['time'] + 24
            df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-24 08:14:00' )

    df['time'] = df['time'].dt.round('ms')
    df = df.sort_values('time', ascending = True, kind = 'mergesort')
    #df = df.reset_index(drop=True)
    print(time.process_time() - start)
    df.set_index('time', inplace = True)
    print(df.head())
    #print(df['time'].head())
    return df
    
def soloA(file_path):
    #skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
    df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';')
    cols = df.columns.tolist()
    new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #reorder the columns into the correct order
    df = df[new_cols]
    return df

def soloB(file_path):
    #skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
    df_B = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';')
    cols = df_B.columns.tolist()
    new_cols = [cols[0]] + cols[9:13] + cols[1:9] + cols[13:17]#reorder the columns into the correct order # adding time as first column
    df_B = df_B[new_cols]
    return df_B

def powerspecplot(df, fs, collist):
    
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