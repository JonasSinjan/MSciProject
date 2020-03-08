import scipy.io
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from processing import processing
import scipy.signal as sps
from fast_histogram import histogram1d

def get_burst_data(windows):
    start = time.time()
    if windows:
        os.environ['MFSA_raw'] = 'C:\\Users\\jonas\\MSci-Data'
    else:
        os.environ['MFSA_raw'] = os.path.expanduser('~/Documents/MsciProject/Data')
        
    project_data = os.environ.get('MFSA_raw')
    flight_data_path = os.path.join(project_data, 'BurstSpectral.mat')
    #print(flight_data_path)

    mat = scipy.io.loadmat(flight_data_path)
    #print(mat.keys())
    #print(type(mat['ddIBS']))
    #print(mat['ddIBS'].shape)
    void_arr = mat['ddOBS'][0][0] #plot obs on top of ibs to show deviation more clearly
    void_ibs = mat['ddIBS'][0][0]
    timeseries = void_arr[9]
    ibs_timeseries = void_ibs[9]
    #print(ibs_timeseries.shape)
    
    y = timeseries[:,0] #x
    y1 = timeseries[:,1] #y
    y2 = timeseries[:,2] #z 
    y3 = timeseries[:,3] #total B  OBS
    ibs_y = ibs_timeseries[:,0] #x
    ibs_y1 = ibs_timeseries[:,1] #y
    ibs_y2 = ibs_timeseries[:,2] #z 
    ibs_y3 = ibs_timeseries[:,3] #total B IBS

    #print(np.sqrt(y[0]**2 + y1[0]**2 + y2[0]**2), y3[0]) - confirms suspicion 4th column is B mag
    #print(np.sqrt(ibs_y[0]**2 + ibs_y1[0]**2 + ibs_y2[0]**2), ibs_y3[0])

    x = [round(x/128,3) for x in range(len(y))] #missing y data
   
    dict_d = {'OBS_X': y, 'OBS_Y': y1, 'OBS_Z': y2, 'OBS_MAGNITUDE': y3, 'IBS_X': ibs_y, 'IBS_Y': ibs_y1, 'IBS_Z': ibs_y2, 'IBS_MAGNITUDE': ibs_y3 }
    df = pd.DataFrame(data=dict_d, dtype = np.float64)
    end = datetime(2020,3,3,15,58,46) + timedelta(seconds = 42463, microseconds=734375)
    
    date_range = pd.date_range(start = datetime(2020,3,3,15,58,46,0), end = end, freq='7812500ns') #1/128 seconds exactly for 1/16 just need microseconds 'ms'
    df.set_index(date_range[:-1], inplace=True) #for some reason, one extra time created
    print(df.head())

    print('df successfully loaded\nExecution time: ', round(time.time() - start,3), ' seconds')

    return df
    

def plot_burst(df):
    x = [x/(128*3600) for x in df.index] #128 vectors a second
    fig = plt.figure()
    plt.subplot(4,1,1)
    plt.plot(x, df['IBS_X'], label = 'IBS')
    plt.plot(x, df['OBS_X'], 'r', label = 'OBS')
    plt.legend(loc='upper right')
    plt.ylabel('Bx [nT]')
    
    plt.subplot(4,1,2)
    plt.plot(x, df['IBS_Y'], label = 'IBS')
    plt.plot(x, df['OBS_Y'], 'r',label = 'OBS')
    plt.legend(loc='upper right')
    plt.ylabel('By [nT]')
    
    plt.subplot(4,1,3)
    plt.plot(x, df['IBS_Z'], label = 'IBS')
    plt.plot(x, df['OBS_Z'], 'r', label = 'OBS')
    plt.ylabel('Bz [nT]')
    plt.legend(loc='upper right')

    plt.subplot(4,1,4)
    plt.plot(x, df['IBS_MAGNITUDE'], label = 'IBS')
    plt.plot(x, df['OBS_MAGNITUDE'], 'r', label = 'OBS')
    plt.ylabel('B [nT]')
    plt.xlabel('Time [Hours]')
    plt.legend(loc='upper right')
    
    plt.suptitle('Magnetic Field with means removed')
    plt.show()


def burst_powerspectra(df, OBS):
    if OBS:
        collist = ['Time', 'OBS_X', 'OBS_Y', 'OBS_Z']
        name_str = 'OBS_burst'
    else:
        collist = ['Time', 'IBS_X', 'IBS_Y', 'IBS_Z']
        name_str = 'IBS_burst'

    processing.powerspecplot(df, 128, collist, False, probe = 'MAG', inst = name_str, inflight = True)


def spectrogram(df, OBS, *, downlimit = 0, uplimit=0.001):
    if OBS:
        collist = ['Time', 'OBS_X', 'OBS_Y', 'OBS_Z']
        name_str = 'OBS_burst'
    else:
        collist = ['Time', 'IBS_X', 'IBS_Y', 'IBS_Z']
        name_str = 'IBS_burst'
        #x = np.sqrt(df[self.collist[1]]**2 + self.df[self.collist[2]]**2 + self.df[self.collist[3]]**2)
    y = (df[collist[1]] + df[collist[2]] + df[collist[3]])
    dflen = len(df)
    div = (dflen)/1000
    #f, Pxx = sps.periodogram(x,fs)
    #div = 500
    fs = 128
    nff = dflen//div
    wind = sps.hamming(int(dflen//div))
    f, t, Sxx = sps.spectrogram(y, fs, window=wind, noverlap = int(dflen//(2*div)), nfft = nff)#,nperseg=700)
    print(type(Sxx))
    #plt.figure()
    #
    #plt.hist(Sxx)
    ax = plt.figure()
    #Sxx = np.where(Sxx<5)
    normalize = mpl.colors.Normalize(vmin=downlimit, vmax=uplimit,  clip = True)
    lognorm = mpl.colors.LogNorm(vmin=downlimit, vmax = uplimit, clip=True)
    plt.pcolormesh(t, f, Sxx, norm = normalize, cmap = 'viridis') #sqrt? 
    #plt.pcolormesh(t, f, Sxx, clim = (0,uplimit))
    plt.semilogy()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    
    plt.title(f'MAG {name_str} Spectrogram @ {fs}Hz, {df.index[0].date()}')
    plt.ylim((10**0,fs/2))
    #plt.clim()
    fig = plt.gcf()
    ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.5, 1]
    if OBS:
        ticks = [0, 10e-6,10e-5,10e-4,10e-3,10e-2]
    cbar = plt.colorbar(ticks = ticks)
    #cbar.ax.set_yticklabels(fontsize=8)
    cbar.set_label('Normalised Power Spectral Density of the Trace')#, rotation=270)
    plt.show()


def get_df_between_seconds(df, start, end):

    time_1 = timedelta(seconds = start)
    time_2 = timedelta(seconds = end)
    
    time_start = pd.to_datetime(df.index[0], infer_datetime_format=True)
    time_1 = time_start + time_1 
    time_2 = time_start + time_2
    df2 = df.between_time(time_1.time(), time_2.time())

    return df2


def heater_data(windows):
    if windows:
        os.environ['MFSA_raw'] = 'C:\\Users\\jonas\\MSci-Data'
    else:
        os.environ['MFSA_raw'] = os.path.expanduser('~/Documents/MsciProject/Data')
        
    project_data = os.environ.get('MFSA_raw')
    flight_data_path = os.path.join(project_data, 'HeaterData.mat')
    print(flight_data_path)

    mat = scipy.io.loadmat(flight_data_path)
    print(mat.keys())

    heater = mat['ddOBS'][0][0]
    print(len(heater))

    timeseries = heater[9]
    y = timeseries[:,0]
    y1 = timeseries[:,1]
    y2 = timeseries[:,2]
    x = range(len(y))
    x = [x/16 for x in x] #16 vectors a second
    fig = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x, y)
    plt.ylabel('B [nT]')
    plt.subplot(3,1,2)
    plt.plot(x, y1)
    plt.ylabel('B [nT]')
    plt.subplot(3,1,3)
    plt.plot(x, y2)
    plt.ylabel('B [nT]')
    plt.xlabel('Time [s]')
    plt.suptitle('Magnetic Field with means removed')
    plt.show()

    heater_cur = mat['Heater'][0][0]
    print(heater_cur)

    plt.figure()
    x = [x/3600 for x in range(len(heater_cur[-1]))]
    plt.plot(x, heater_cur[-1])
    plt.ylabel('Current [A]')
    plt.xlabel('Time [Hours]')
    plt.show()

if __name__ == "__main__":
    windows = True
    df = get_burst_data(windows)
    OBS = False

    df2 = get_df_between_seconds(33000, 33400)

    spectrogram(df2, OBS, downlimit = 0, uplimit=0.001)
    #burst_powerspectra(df2, OBS)

    """
    collist = ['Time', 'IBS_X', 'IBS_Y', 'IBS_Z']
    y = (df[collist[1]] + df[collist[2]] + df[collist[3]])
    dflen = len(df)
    div = (dflen)/1000
    fs = 128
    wind = sps.hamming(int(dflen//div))
    nff = dflen//div
    f, t, Sxx = sps.spectrogram(y, fs, window=wind, noverlap = int(dflen//(2*div)), nfft = nff)#,nperseg=700)
    print(type(Sxx), Sxx.shape)
    plt.hist(Sxx.flatten(), histtype='step', bins = 50)
    plt.show()
    Sxx_5 = np.where(Sxx<5)
    plt.hist(Sxx_5, histtype='step', bins = 50)
    plt.show()
    Sxx_10 = np.where(Sxx<10)
    plt.hist(Sxx_10, histtype='step', bins = 50)
    plt.show()  
    #plt.hist(Sxx, bins = 50)
    """