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
from collections import defaultdict

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
        #print(mfsa_init_path)
        if self.day == 1:
            mfsa_init_path = os.path.join(mfsa_init_path, 'day_one')
        elif self.day == 2:
            mfsa_init_path = os.path.join(mfsa_init_path, 'day_two')
        #print(mfsa_init_path)
        mfsa_fol_A = os.path.join(mfsa_init_path, 'A')
        mfsa_fol_B = os.path.join(mfsa_init_path, 'B')
        #print(mfsa_fol_A)
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

        print(self.df.head())
        print(self.df.tail())

    def spectrogram(self, *, downlimit = 0, uplimit=0.1):
        x = np.sqrt(self.df[self.collist[1]]**2 + self.df[self.collist[2]]**2 + self.df[self.collist[3]]**2)
        y = (self.df[self.collist[1]] + self.df[self.collist[2]] + self.df[self.collist[3]])
        div = (self.dflen)/1000
        #f, Pxx = sps.periodogram(x,fs)
        #div = 500
        nff = self.dflen//div
        wind = sps.hamming(int(self.dflen//div))
        f, t, Sxx = sps.spectrogram(y, self.fs, window=wind, noverlap = int(self.dflen//(2*div)), nfft = nff)#,nperseg=700)
        ax = plt.figure()
        normalize = mpl.colors.Normalize(vmin=downlimit, vmax=uplimit,  clip = True)
        plt.pcolormesh(t, f, Sxx, norm = normalize, cmap = 'viridis') #sqrt? 
        #plt.pcolormesh(t, f, Sxx, clim = (0,uplimit))
        plt.semilogy()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        if hasattr(self, 'name'):
            plt.title(f'{self.name} - Probe {self.probe} @ {self.fs}Hz, {self.start.date()}')
        else:
            plt.title(f'Probe {self.probe} @ {self.fs}Hz, {self.start.date()}')
        plt.ylim((10**0,self.fs/2))
        #plt.clim()
        fig = plt.gcf()
        cbar = plt.colorbar(ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.5, 1])
        #cbar.ax.set_yticklabels(fontsize=8)
        cbar.set_label('Normalised Power Spectral Density of the Trace')#, rotation=270)
        
        plt.show()

    def powerspectra(self):
        if hasattr(self, 'name'):
            processing.powerspecplot(self.df, self.fs, self.collist, alt=False, inst = self.name, save = False)
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
        
        plt.xlabel('Time [H:M:S]')
        plt.ylabel('dB [nT]')
        plt.title(f'Probe {self.probe} @ {self.fs}Hz, {self.start.date()}')
        plt.legend(loc="best")
        plt.show()

    def moving_powerfreq(self, OBS, len_of_sections = 600, desired_freqs = [8.0], *, scaling = 'spectrum'):
        
        probe_x = self.collist[1]
        probe_y = self.collist[2]
        probe_z = self.collist[3]
        #mag = collist[3]

        #self.df = self.df[collist]

        def get_powerspec_of_desired_freq(f, Pxx, desired_frequencies):
            assert type(desired_frequencies) == list
            dfreq = 0.02
            mean_power_dict = defaultdict(list)

            for i in desired_frequencies:
                #print(i)
                index_tmp = np.where((f >= i - dfreq/2 ) & (f <= i + dfreq/2))
                #print(index_tmp)
                mean_power = max(Pxx[index_tmp])
                #print(mean_power)
                mean_power_dict[str(i)] = [mean_power]
                #print(mean_power_dict)

            return mean_power_dict

        sections = len(self.df)//(self.fs*len_of_sections)
        start = 0
        end = len_of_sections*self.fs

        for i in range(sections):
            if end >= len(self.df):
                end = len(self.df)
            df_tmp = self.df.iloc[start:end,:]
            x = df_tmp[probe_x]#[:20000]
            f_x, Pxx_x = sps.periodogram(x, self.fs, scaling = f'{scaling}')
            x_y = df_tmp[probe_y]#[:20000]
            f_y, Pxx_y = sps.periodogram(x_y, self.fs, scaling = f'{scaling}')
            x_z = df_tmp[probe_z]#[:20000]
            f_z, Pxx_z = sps.periodogram(x_z, self.fs, scaling = f'{scaling}')
            x_t = x + x_y + x_z #trace
            f_t, Pxx_t = sps.periodogram(x_t, self.fs, scaling = f'{scaling}')
            #x_m = df_tmp[mag]
            #f_m, Pxx_m = sps.periodogram(x_m, self.fs, scaling = f'{scaling}')

            if i == 0:
                x_dict = get_powerspec_of_desired_freq(f_x, Pxx_x, desired_freqs)
                y_dict = get_powerspec_of_desired_freq(f_y, Pxx_y, desired_freqs)
                z_dict = get_powerspec_of_desired_freq(f_z, Pxx_z, desired_freqs)
                t_dict = get_powerspec_of_desired_freq(f_t, Pxx_t, desired_freqs)
                #m_dict = get_powerspec_of_desired_freq(f_m, Pxx_m, desired_freqs)
                #print(type(x_dict))
            else:
                x_dict_tmp = get_powerspec_of_desired_freq(f_x, Pxx_x, desired_freqs)
                y_dict_tmp = get_powerspec_of_desired_freq(f_y, Pxx_y, desired_freqs)
                z_dict_tmp = get_powerspec_of_desired_freq(f_z, Pxx_z, desired_freqs)
                t_dict_tmp = get_powerspec_of_desired_freq(f_t, Pxx_t, desired_freqs)
                #m_dict_tmp = get_powerspec_of_desired_freq(f_m, Pxx_m, desired_freqs)

                for j in desired_freqs:
                    x_dict[str(j)].append(x_dict_tmp[str(j)][0])
                    y_dict[str(j)].append(y_dict_tmp[str(j)][0])
                    z_dict[str(j)].append(z_dict_tmp[str(j)][0])
                    t_dict[str(j)].append(t_dict_tmp[str(j)][0])
                    #m_dict[str(j)].append(m_dict_tmp[str(j)][0])
            
            start += len_of_sections
            end += len_of_sections

        #print(x_dict[str(8.0)])
        
        plt.figure()
        #plt.plot(range(sections), x_dict[str(8.0)], label = 'X')
        #plt.plot(range(sections), y_dict[str(8.0)], label = 'Y')
        #plt.plot(range(sections), z_dict[str(8.0)], label = 'Z')
        x = [i*len_of_sections/3600 for i in range(sections)]
        for j in desired_freqs:
            plt.plot(x, t_dict[str(j)], label = f'T - {j}Hz')
            print(max(t_dict[str(j)]))
            #plt.plot(x, t_dict[str(16.667)], label = 'T - 16.667Hz')
        #plt.plot(x, t_dict[str(16.0)], label = 'T - 16Hz')
        #plt.plot(x, t_dict[str(0.119)], label = 'T - 0.119Hz')
        #plt.plot(x, t_dict[str(0.238)], label = 'T - 0.238Hz')
        #plt.plot(x, t_dict[str(0.357)], label = 'T - 0.357Hz')
        #plt.plot(x, t_dict[str(0.596)], label = 'T - 0.596Hz')
        
        #plt.plot(range(sections), m_dict[str(8.0)], label = 'M')
        plt.legend(loc='upper right')
        plt.ylabel('Power [dB]')
        plt.xlabel('Time [Hours]')
        plt.title(f'{len_of_sections//60} min moving max Power')
        plt.semilogy()
        plt.show()


if __name__ == "__main__":
    """
    METIS - 10:10-10:56
    EUI - 9:24-10:09
    SPICE - 10:57-11:18
    STIX - 11:44-12:17
    SWA - 12:18-13:52
    PHI - 8:05-8:40
    SoloHI - 11:19-11;44
    EPD - 14:43-14:59 #be wary as epd in different regions #full ==>13:44-14:58
    """
    day = 2
    probe = 7 #doing only 7,9,10 (7 closest to instruments, 9 at mag ibs, 10 at mag obs)
    sampling_fs = 100
    
    """
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(0.05)
    """

    daytwo = mfsa_object(day, datetime(2019,6,24,7,27), datetime(2019,6,24,15,0), probe, sampling_fs, timezone = 'MAG', name = 'Full_Day_2')
    daytwo.get_data()
    daytwo.spectrogram(uplimit = 0.1)
    #daytwo.powerspectra()

    #eui = mfsa_object(day, datetime(2019,6,24,9,24), datetime(2019,6,24,10,9), probe, sampling_fs, timezone = 'MAG', name = 'EUI')
    #eui.get_data()
    #eui.plot()
    #eui.moving_powerfreq(True, len_of_sections=60, desired_freqs=[8.0, 16.667])
    
    #eui.spectrogram()
    #eui.powerspectra()


    #daytwo = mfsa_object(day, datetime(2019,6,24,7,27), datetime(2019,6,24,15,0), probe, sampling_fs, timezone = 'MAG', name = 'Full_Day_2')
    #daytwo.get_data()
    #daytwo.moving_powerfreq(True, len_of_sections=60, desired_freqs=[8.0, 16.667], scaling='density')
    #daytwo.moving_powerfreq(True, len_of_sections=60, desired_freqs=[8.0, 16.667], scaling='spectrum')

    #daytwo.moving_powerfreq(True, len_of_sections=100, desired_freqs=[8.0, 16.667], scaling='density')
    #daytwo.moving_powerfreq(True, len_of_sections=100, desired_freqs=[8.0, 16.667], scaling='spectrum')
    #daytwo.spectrogram(downlimit = 0.5, uplimit = 1.0)
    #daytwo.powerspectra()

    #metis = mfsa_object(day,datetime(2019,6,24,10,10), datetime(2019,6,24,10,56), probe, sampling_fs, timezone = 'MAG', name = 'METIS')
    #metis.get_data()
    #metis.spectrogram(downlimit = 0, uplimit = 0.1) #need 0.4 for trace, 0.1 for absolute magnitude
    #metis.powerspectra()