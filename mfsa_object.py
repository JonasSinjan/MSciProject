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
from plot_raw_current import plot_raw
from magplot_2 import mag
import scipy.optimize as spo
import seaborn as sns


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
        start = time.time()
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
        start_csv, end_csv = processing.which_csvs(soloA_bool, day, self.start, self.end, tz_MAG=False) #this function (in processing.py) finds the number at the end of the csv files we want
        print(start_csv, end_csv)

        all_files = [0]*(end_csv + 1 - start_csv)
        
        if self.day == 2:
            for index, i in enumerate(range(start_csv, end_csv + 1)): #this will loop through and add the csv files that contain the start and end time set above
                if soloA_bool:
                    all_files[index] = os.path.join(mfsa_fol_A, f'SoloA_2019-06-24--08-14-46_{i}.csv')
                    
                else:
                    all_files[index] = os.path.join(mfsa_fol_B, f'SoloB_2019-06-24--08-14-24_{i}.csv')
        
        elif self.day == 1:
            for index, i in enumerate(range(start_csv, end_csv + 1)): #this will loop through and add the csv files that contain the start and end time set above
                if soloA_bool:
                    all_files[index] = os.path.join(mfsa_fol_A, f'SoloA_2019-06-21--08-10-10_{i}.csv')
                    
                else:
                    all_files[index] = os.path.join(mfsa_fol_B, f'SoloB_2019-06-21--08-09-10_{i}.csv')

        if soloA_bool:
            self.df = processing.read_files(all_files, soloA_bool, self.fs, self.collist, self.day, start_dt = self.start, end_dt = self.end)
            if self.day == 2:
                rotate_mat = processing.rotate_24(soloA_bool)[self.probe-1]
            elif self.day == 1:
                rotate_mat = processing.rotate_21(soloA_bool)[self.probe-1]
        else:
            self.df = processing.read_files(all_files, soloA_bool, self.fs, self.collist, self.day, start_dt = self.start, end_dt = self.end)
            if self.day == 2:
                rotate_mat = processing.rotate_24(soloA_bool)[self.probe-9]
            elif self.day == 1:
                rotate_mat = processing.rotate_21(soloA_bool)[self.probe-9]

        self.df.iloc[:,0:3] = np.matmul(rotate_mat, self.df.iloc[:,0:3].values.T).T
        #find the df of the exact time span desired
        self.df = self.df.between_time(self.start.time(), self.end.time())
        self.df = processing.shifttime(self.df, soloA_bool, self.day)
        self.dflen = len(self.df) 

        print(self.df.head())
        print(self.df.tail())

        print('Data Loaded - Execution Time: ' ,round(time.time()-start, 2), 'seconds')

    def spectrogram(self, *, downlimit = 0, uplimit=0.1, div = 1000, fontsize=18, ylower=10e-1):
        plt.rcParams.update({'font.size': fontsize})
        start = time.time()
        x = np.sqrt(self.df[self.collist[1]]**2 + self.df[self.collist[2]]**2 + self.df[self.collist[3]]**2)
        y = (self.df[self.collist[1]] + self.df[self.collist[2]] + self.df[self.collist[3]])
        #div = (self.dflen)/len_div
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
        #if hasattr(self, 'name'):
            #plt.title(f'{self.name} - Probe {self.probe} @ {self.fs}Hz, {self.start.date()}')
        #else:
            #plt.title(f'Probe {self.probe} @ {self.fs}Hz, {self.start.date()}')
        plt.ylim((ylower,self.fs/2))
        #plt.clim()
        fig = plt.gcf()
        cbar = plt.colorbar(ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.5, 1])
        #cbar.ax.set_yticklabels(fontsize=8)
        cbar.set_label('Normalised Power Spectral Density of the Trace')#, rotation=270)
        print('Spectrogram Created - Execution Time: ' ,round(time.time()-start, 2), 'seconds')
        plt.show()

    def powerspectra(self):
        if hasattr(self, 'name'):
            processing.powerspecplot(self.df, self.fs, self.collist, alt=False, inst = self.name, save = False)
        else:
            processing.powerspecplot(self.df, self.fs, self.collist, alt=False, save = False)

    def plot(self):
        df2 = self.df.resample('2s').mean()
        print(df2.head())
        plt.figure()
        #tmp = []
        """
        i = 0
        day_1_offsets = [203, -49, 102.5] #p9
        day_1_offsets = [26, 14, 31] #p8
        day_1_offsets = [5,-45, 61] #p10
        day_1_offsets = [-65,-38, -5] #p10 non rotated
        day_1_offsets = [4.4,47.8,2.2] p7
        
        plt.plot(df2.index.time, df2[self.collist[2]]-49, label ='Probe09_Y')
        for col in self.collist[1:]:
           df2[col] = df2[col] + day_1_offsets[i]#- df2[col].mean()
           plt.plot(df2.index.time, df2[col], label=str(col))
           i+=1
       
            #var_1hz = np.std(df2[col])
            #var_1khz = np.std(df2[col])
            #print('std - 1Hz', col, var_1hz)
            #print('std - 1kHz', col,  var_1khz)
            #tmp.append(var_1hz)
            #tmp.append(var_1khz)
            #print(df2[col].abs().idxmax())
            #b_magnitude += df2[col]**2
        #plt.plot(df2.index.time, b_magnitude, label = '|B|')
        """
        #to create same plot as p10 airbus, p9 ours
        
        #plt.plot(df2.index.time, df2[self.collist[2]]-49, label = 'Y')
        #plt.plot(df2.index.time, -(df2[self.collist[1]]+203), label = 'Z')
        #plt.plot(df2.index.time, (df2[self.collist[3]]+102.5), label = 'X')

        #b_magnitude = np.sqrt((df2[self.collist[2]]-49)**2 + (df2[self.collist[1]]+203)**2 + (df2[self.collist[3]]+102.5)**2)
        #b_magnitude = np.sqrt((df2[self.collist[2]]-45)**2 + (df2[self.collist[1]]+5)**2 + (df2[self.collist[3]]+61)**2)
        #b_magnitude = np.sqrt((df2[self.collist[2]]+47.8)**2 + (df2[self.collist[1]]+4.4)**2 + (df2[self.collist[3]]+2.2)**2)#p7
        b_magnitude = np.sqrt((df2[self.collist[2]]-46.7)**2 + (df2[self.collist[1]]+6)**2 + (df2[self.collist[3]]+63.5)**2)
        plt.plot(df2.index.time, b_magnitude, label = '|B|')
        
        y = np.linspace(-0.25,10,100)

        if self.day == 1:
            plt.plot([datetime(2019,6,21,7,41,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,7,31,0).time(), y = -0.5, s = 'Core On')

            plt.plot([datetime(2019,6,21,8,59,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,8,59,0).time(), y = -0.5, s = 'Power Amplifier')
            
            plt.plot([datetime(2019,6,21,8,56,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,8,45,0).time(), y = -0.5, s = 'MAG')

            plt.plot([datetime(2019,6,21,9,58,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,9,56,0).time(), y = -0.5, s = 'STIX')

            plt.plot([datetime(2019,6,21,10,20,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,10,18,0).time(), y = -0.5, s = 'EPD')

            plt.plot([datetime(2019,6,21,10,35,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,10,40,0).time(), y = -0.5, s = 'SWA')

            plt.plot([datetime(2019,6,21,11,46,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,11,38,0).time(), y = -0.5, s = 'METIS')

            plt.plot([datetime(2019,6,21,12,10,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,12,5,0).time(), y = -0.5, s = 'EUI')

            plt.plot([datetime(2019,6,21,12,32,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,12,27,0).time(), y = -0.5, s = 'PHI')

            plt.plot([datetime(2019,6,21,12,40,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,12,34,0).time(), y = -0.5, s = 'SPICE')

            plt.plot([datetime(2019,6,21,12,49,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,12,48,0).time(), y = -0.5, s = 'SoloHI')

            plt.plot([datetime(2019,6,21,14,4,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,21,13,44,0).time(), y = -0.5, s = 'Power Amplifier')

        elif self.day == 2:
            #plt.plot([datetime(2019,6,24,6,24,0).time() for i in y], y, linestyle="--", color = 'black')
            #plt.text(x = datetime(2019,6,24,6,24,0).time(), y = -0.5, s = 'Core On')

            plt.plot([datetime(2019,6,24,8,2,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,24,8,1,0).time(), y = -0.5, s = 'Power Amplifier')
            
            plt.plot([datetime(2019,6,24,7,28,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,24,7,28,0).time(), y = -0.5, s = 'MAG')

            plt.plot([datetime(2019,6,24,11,41,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,24,11,40,0).time(), y = -0.5, s = 'STIX')

            plt.plot([datetime(2019,6,24,13,42,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,24,13,41,0).time(), y = -0.5, s = 'EPD')

            plt.plot([datetime(2019,6,24,12,15,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,24,12,14,0).time(), y = -0.5, s = 'SWA')

            plt.plot([datetime(2019,6,24,10,9,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,24,10,8,0).time(), y = -0.5, s = 'METIS')

            plt.plot([datetime(2019,6,24,9,22,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,24,9,21,0).time(), y = -0.5, s = 'EUI')

            plt.plot([datetime(2019,6,24,8,8,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,24,8,1,0).time(), y = -0.5, s = 'PHI')

            plt.plot([datetime(2019,6,24,8,41,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,24,8,35,0).time(), y = -0.5, s = 'SPICE')

            plt.plot([datetime(2019,6,24,11,16,0).time() for i in y], y, linestyle="--", color = 'black')
            plt.text(x = datetime(2019,6,24,11,16,0).time(), y = -0.5, s = 'SoloHI')

            #plt.plot([datetime(2019,6,24,15,0,0).time() for i in y], y, linestyle="--", color = 'black')
            #plt.text(x = datetime(2019,6,24,15,0,0).time(), y = -0.5, s = 'Power Amplifier')

        plt.ylim(-1,7)

        plt.xlabel('Time [H:M:S]')
        plt.ylabel('dB [nT]')
        plt.title(f'Probe {self.probe} @ {self.fs}Hz, {self.start.date()}')
        plt.legend(loc="best")
        plt.show()

    def moving_variation(self,*,len_of_sections):

        sections = len(self.df)//(self.fs*len_of_sections)
        start = 0
        end = len_of_sections*self.fs

        x_var_list, y_var_list, z_var_list, mag_var_list = [], [], [], []

        for i in range(sections):
            if end >= len(self.df):
                end = len(self.df)
            df_tmp = self.df.iloc[start:end,:]
            x_var = df_tmp[self.collist[1]].std()
            y_var = df_tmp[self.collist[2]].std()
            z_var = df_tmp[self.collist[3]].std()
            magnitude = np.sqrt(df_tmp[self.collist[1]]**2 +df_tmp[self.collist[2]]**2 + df_tmp[self.collist[3]]**2)
            magnitude_var = magnitude.std()


            x_var_list.append(x_var)
            y_var_list.append(y_var)
            z_var_list.append(z_var)
            mag_var_list.append(magnitude_var)

            start += len_of_sections
            end += len_of_sections

        x = [i*len_of_sections/3600 for i in range(sections)]

        plt.figure()
        plt.plot(x, x_var_list, label = 'X VAR')
        plt.plot(x, y_var_list, label = 'Y VAR')
        plt.plot(x, z_var_list, label = 'Z VAR')
        plt.plot(x, mag_var_list, label = '|B| VAR')

        plt.legend(loc="best")
        plt.show()

    def plot_B_v_I(self):
        if self.name in ['EUI', 'METIS', 'PHI', 'SWA', 'SoloHI', 'STIX', 'SPICE', 'EPD']:
            df2 = self.df.resample('1s').mean()
            
            if self.day == 1:
                if self.probe < 8:
                    timezone_change = pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 30, milliseconds = 137)
                else:
                    timezone_change = pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 1, milliseconds = 606)
            elif self.day == 2:
                if self.probe < 8:
                    timezone_change = pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)
                else:
                    timezone_change = pd.Timedelta(days = 0, hours = 1, minutes = 58, seconds = 46, milliseconds = 499)

            start = self.start - timezone_change
            end = self.end - timezone_change

            current_df = plot_raw(True, self.name, self.day, plot=False)
            current_df = current_df.between_time(start.time(), end.time())
            current_df = current_df.resample('1s').mean()

            print(len(current_df), len(df2))
            df2 = df2.iloc[:min(len(current_df), len(df2))]
            current_df = current_df.iloc[:min(len(current_df), len(df2))]
            print(current_df.head())

            xdata = current_df[f'{self.name} Current [A]']
            def line(x,a,b):
                return a*x + b

            for col in self.collist[1:]:
                df2[col] = df2[col] - df2[col].mean()

            params_x,cov_x = spo.curve_fit(line, xdata, df2[self.collist[1]])
            params_y,cov_y = spo.curve_fit(line, xdata, df2[self.collist[2]])
            params_z,cov_z = spo.curve_fit(line, xdata, df2[self.collist[3]])

            perr_x = np.sqrt(np.diag(cov_x))
            perr_y = np.sqrt(np.diag(cov_y))
            perr_z = np.sqrt(np.diag(cov_z))
            
            plt.figure()

            #plt.plot(xdata, params_x[0]*xdata + params_x[1], 'b-',label=f'X {round(params_x[0],2)} +/-{round(perr_x[0],2)}')
            #plt.plot(xdata, params_y[0]*xdata + params_y[1], color = 'orange', linestyle = '-',label=f'Y {round(params_y[0],2)} +/-{round(perr_y[0],2)}')
            #plt.plot(xdata, params_z[0]*xdata + params_z[1], 'g-',label=f'Z {round(params_z[0],2)} +/-{round(perr_z[0],2)}')

            concatenate = pd.concat([current_df, df2], axis = 1)
            #concatenate.head()
           
            for col in self.collist[1:]:
                #plt.scatter(current_df[f'{self.name} Current [A]'], df2[col])
                sns.scatterplot(x=current_df[f'{self.name} Current [A]'], y = f'{col}', data = concatenate, label = f'{col}')
            plt.title(f'{self.name} - Probe {self.probe} @ 1Hz, {self.start.date()}')

            plt.xlabel('Current [A]')
            plt.ylabel('dB [nT]')
            plt.legend(loc="best")
            plt.show()
        else:
            print('The objects attribute: name is not a correct instrument.')

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
    #start = time.time()
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
    probe = 7 #doing only 7,9,10 (7 closest to instruments, 9 at mag ibs, 10 at mag obs) #9 is actually 10 and 10 is actually 9
    sampling_fs = 100
    
    """
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(0.05)
    """

    #daytwo = mfsa_object(day, datetime(2019,6,24,7,27), datetime(2019,6,24,15,0), probe, sampling_fs, timezone = 'MAG', name = 'Full_Day_2')
    #daytwo.get_data()
    #daytwo.spectrogram(uplimit = 0.1)
    #daytwo.powerspectra()

    #eui = mfsa_object(day, datetime(2019,6,24,9,24), datetime(2019,6,24,10,9), probe, sampling_fs, timezone = 'MAG', name = 'EUI')
    #eui.get_data()
    #eui.plot_B_v_I()
    #eui.plot()
    #eui.moving_powerfreq(True, len_of_sections=60, desired_freqs=[8.0, 16.667])
    
    #eui.spectrogram()
    #eui.powerspectra()


    #daytwo = mfsa_object(day, datetime(2019,6,24,6,30), datetime(2019,6,24,15,0), probe, sampling_fs, timezone = 'MAG', name = 'Full_Day_2') #7:27 normally start, 15:00 end
    #daytwo.get_data()
    #daytwo.plot()
    #daytwo.moving_powerfreq(True, len_of_sections=60, desired_freqs=[8.0, 16.667], scaling='density')
    #daytwo.moving_powerfreq(True, len_of_sections=60, desired_freqs=[8.0, 16.667], scaling='spectrum')

    #daytwo.moving_powerfreq(True, len_of_sections=100, desired_freqs=[8.0, 16.667], scaling='density')
    #daytwo.moving_powerfreq(True, len_of_sections=100, desired_freqs=[8.0, 16.667], scaling='spectrum')
    #daytwo.spectrogram(downlimit = 0, uplimit = 0.1, div = 1500, fontsize = 18, ylower = 10e-1)
    #daytwo.powerspectra()

    metis = mfsa_object(day,datetime(2019,6,24,10,10), datetime(2019,6,24,10,56), probe, sampling_fs, timezone = 'MAG', name = 'METIS')
    metis.get_data()
    #metis.spectrogram(downlimit = 0, uplimit = 0.1) #need 0.4 for trace, 0.1 for absolute magnitude
    metis.powerspectra()

    #dayone = mfsa_object(day, datetime(2019,6,21,6,30,0), datetime(2019,6,21,14,45,0), probe, sampling_fs, timezone = 'MAG', name = 'Full_Day_1')
    #dayone.get_data()
    #dayone.plot()
    #dayone.moving_variation(len_of_sections=120)
    #dayone.spectrogram(downlimit = 0, uplimit = 0.1, div = 1000)
    #print('Execution Time: ' ,round(time.time()-start, 2), 'seconds')