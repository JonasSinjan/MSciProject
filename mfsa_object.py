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
from current import current_peaks

class mfsa_object:

    def __init__(self, day, start, end, probe, sampling_freq, *, timezone='MFSA', name=None):
        assert day == 1 or day == 2
        assert 0 < probe < 13
        assert sampling_freq <= 1000
        assert timezone == 'MFSA' or timezone == 'MAG'
        self.day = day #int 1 or 2
        
        if timezone== 'MFSA': #mfsa, local German Time
            self.start = start #must be datetime format
            self.end = end #must be datetime format
        elif timezone == 'MAG': #spacecraft time
            if day == 1:
                if probe < 8:
                    timezone_change = pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 30, milliseconds = 137)
                else:
                    timezone_change = pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 1, milliseconds = 606)
            elif day == 2:
                if probe < 8:
                    timezone_change = pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)
                else:
                    timezone_change = pd.Timedelta(days = 0, hours = 1, minutes = 58, seconds = 46, milliseconds = 499)

            self.start = start + timezone_change 
            self.end = end + timezone_change

        self.probe = probe #int
        self.fs = sampling_freq #int
        if name != None:
            self.name = name #str

    def get_data(self, windows=True):
        if windows:
            os.environ['MFSA_raw'] = 'C:\\Users\\jonas\\MSci-Data'
        else:
            os.environ['MFSA_raw'] = os.path.expanduser('~/Documents/MsciProject/Data')

        mfsa_init_path = os.environ.get('MFSA_raw')
        print(mfsa_init_path)
        if self.day == 1:
            mfsa_init_path = os.path.join(mfsa_init_path, 'day_one')
        elif self.day == 2:
            mfsa_init_path = os.path.join(mfsa_init_path, 'day_two')
        print(mfsa_init_path)
        mfsa_fol_A = os.path.join(mfsa_init_path, 'A')
        mfsa_fol_B = os.path.join(mfsa_init_path, 'B')
        print(mfsa_fol_A)
        if self.probe < 9:
            soloA_bool = True
        else:
            soloA_bool = False
        if self.probe < 10:
            num_str = f'0{self.probe}'
        else: 
            num_str = self.probe

        self.collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']

        #finding the correct MFSA data files
        start_csv, end_csv = processing.which_csvs(soloA_bool, 2, self.start, self.end, tz_MAG=False) #this function (in processing.py) finds the number at the end of the csv files we want
        print(start_csv, end_csv)

        all_files = [0]*(end_csv + 1 - start_csv)
        
        for index, i in enumerate(range(start_csv, end_csv + 1)): #this will loop through and add the csv files that contain the start and end time set above
            if soloA_bool:
                all_files[index] = os.path.join(mfsa_fol_A, f'SoloA_2019-06-24--08-14-46_{i}.csv')
                
            else:
                all_files[index] = os.path.join(mfsa_fol_B, f'SoloB_2019-06-24--08-14-24_{i}.csv')

        if soloA_bool:
            self.df = processing.read_files(all_files, soloA_bool, self.fs, self.collist, self.day, start_dt = self.start, end_dt = self.end)
            rotate_mat = processing.rotate_24(soloA_bool)[self.probe-1]
        else:
            self.df = processing.read_files(all_files, soloA_bool, self.fs, self.collist, self.day, start_dt = self.start, end_dt = self.end)
            rotate_mat = processing.rotate_24(soloA_bool)[self.probe-9]

        self.df.iloc[:,0:3] = np.matmul(rotate_mat, self.df.iloc[:,0:3].values.T).T
        #find the df of the exact time span desired
        self.df = self.df.between_time(self.start.time(), self.end.time())
        self.df = processing.shifttime(self.df, soloA_bool, 2)
        self.dflen = len(self.df) 

    def spectrogram(self, *, uplimit=0.1):
        x = np.sqrt(self.df[self.collist[1]]**2 + self.df[self.collist[2]]**2 + self.df[self.collist[3]]**2)
        div = (self.dflen)/1000
        #f, Pxx = sps.periodogram(x,fs)
        #div = 500
        nff = self.dflen//div
        wind = sps.hamming(int(self.dflen//div))
        f, t, Sxx = sps.spectrogram(x,self.fs, window=wind, noverlap = int(self.dflen//(2*div)), nfft = nff)#,nperseg=700)
        ax = plt.figure()
        plt.pcolormesh(t, f, Sxx, vmin = 0.,vmax = uplimit)
        plt.semilogy()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(f'Spectrogram: Probe {self.probe} @ {self.fs}Hz, {self.start.date()}')
        plt.ylim((10**0,self.fs/2))
        plt.clim()
        fig = plt.gcf()
        cbar = plt.colorbar()
        #cbar.ax.set_yticklabels(fontsize=8)
        cbar.set_label('Normalised Power/Frequency')#, rotation=270)  

        #fig, ax2 = plt.subplots()
        #Pxx, freqs, bins, im = ax2.specgram(x, Fs=sampling_freq)#, noverlap=900)
        #ax2.set_yscale('log')
        #ax2.set_ylim((10**0,sampling_freq/2))

        plt.show()

    def powerspectra(self):
        if hasattr(self, 'name'):
            processing.powerspecplot(self.df, self.fs, self.collist, alt=False, inst = self.name, save = True)
        else:
            processing.powerspecplot(self.df, self.fs, self.collist, alt=False, save = False)

    def plot(self):
        df2 = self.df.resample('1s').mean()
        #tmp = []
        for col in self.collist[1:]:
            df2[col] = df2[col] - df2[col].mean()
            plt.plot(df2.index.time, df2[col], label=str(col))
            
            #var_1hz = np.std(df2[col])
            #var_1khz = np.std(df2[col])
            #print('std - 1Hz', col, var_1hz)
            #print('std - 1kHz', col,  var_1khz)
            #tmp.append(var_1hz)
            #tmp.append(var_1khz)
            #print(df2[col].abs().idxmax())
        
        plt.xlabel('Time (s)')
        plt.ylabel('dB (nT)')
        plt.title(f'Probe {self.probe} @ {self.fs}Hz, {self.start.date()}')
        plt.legend(loc="best")
        plt.show()

if __name__ == "__main__":
    # METIS - 10:10-10:56
    # EUI - 9:24-10:09
    # SPICE - 10:57-11:18
    # STIX - 11:44-12:17
    # SWA - 12:18-13:52
    # PHI - 8:05-8:40
    # SoloHI - 11:19-11;44
    # EPD - 14:43-14:59 #be wary as epd in different regions #full ==>13:44-14:58
    day = 2
    start_dt = datetime(2019,6,24,9,24) #in spacecraft time
    end_dt = datetime(2019,6,24,10,9)
    probe = 9
    sampling_fs = 100

    eui = mfsa_object(day, start_dt, end_dt, probe, sampling_fs, timezone = 'MAG', name = 'EUI')
    eui.get_data()
    eui.plot()