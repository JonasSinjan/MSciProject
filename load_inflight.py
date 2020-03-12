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
from collections import defaultdict
#from fast_histogram import histogram1d

class burst_data:

    def __init__(self):
        pass

    def get_df_from_csv(self, day=1):
        start_time = time.time()

        assert day == 1 or day == 2 or day == 3
        os.environ['MSci-Data'] = 'C:\\Users\\jonas\\MSci-Data'
        project_data = os.environ.get('MSci-Data')
        flight_data_path = os.path.join(project_data, f'burst_data_df_file_2_day_{day}.csv')
        self.df = pd.read_csv(flight_data_path)
        print(self.df.head())

        print('df successfully loaded from csv\nExecution time: ', round(time.time() - start_time,3), ' seconds')

    def get_df_from_mat(self, *, file_one = True, start = 0, end = 128*3600*10):
        start_time = time.time()
        
        os.environ['MSci-Data'] = 'C:\\Users\\jonas\\MSci-Data'
        project_data = os.environ.get('MSci-Data')
        
        if file_one:
            flight_data_path = os.path.join(project_data, 'BurstSpectral.mat')
            end = 5435358
        else:
            flight_data_path = os.path.join(project_data, 'BurstSpectral2.mat')
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
        
        y = timeseries[start:end,0] #x
        #print(len(y)) # file 2 has 72 hours
        y1 = timeseries[start:end,1] #y
        y2 = timeseries[start:end,2] #z 
        y3 = timeseries[start:end,3] #total B  OBS
        ibs_y = ibs_timeseries[start:end,0] #x
        ibs_y1 = ibs_timeseries[start:end,1] #y
        ibs_y2 = ibs_timeseries[start:end,2] #z 
        ibs_y3 = ibs_timeseries[start:end,3] #total B IBS

        #print(np.sqrt(y[0]**2 + y1[0]**2 + y2[0]**2), y3[0]) - confirms suspicion 4th column is B mag
        #print(np.sqrt(ibs_y[0]**2 + ibs_y1[0]**2 + ibs_y2[0]**2), ibs_y3[0])

        #x = [round(x/128,3) for x in range(len(y))] #missing y data
    
        dict_d = {'OBS_X': y, 'OBS_Y': y1, 'OBS_Z': y2, 'OBS_MAGNITUDE': y3, 'IBS_X': ibs_y, 'IBS_Y': ibs_y1, 'IBS_Z': ibs_y2, 'IBS_MAGNITUDE': ibs_y3 }
        df = pd.DataFrame(data=dict_d, dtype = np.float64)
        if file_one:
            end_time = datetime(2020,3,3,15,58,46) + timedelta(seconds = 42463, microseconds=734375)
            date_range = pd.date_range(start = datetime(2020,3,3,15,58,46,0), end = end_time, freq='7812500ns') #1/128 seconds exactly for 1/16 just need microseconds 'ms'
            df.set_index(date_range[:-1], inplace=True) #for some reason, one extra time created
            
        print(df.head())

        print('df successfully loaded\nExecution time: ', round(time.time() - start_time,3), ' seconds')

        self.df = df
        self.fs = 128
        
    def df_to_csv(self, name):
        self.df = self.df.astype({'OBS_X': 'float32', 'OBS_Y': 'float32', 'OBS_Z': 'float32', 'OBS_MAGNITUDE': 'float32', 'IBS_X': 'float32', 'IBS_Y': 'float32', 'IBS_Z': 'float32', 'IBS_MAGNITUDE': 'float32'})
        self.df.to_csv(f'C:\\Users\\jonas\\MSci-Data\\burst_data_df_{name}.csv')

    def get_df_between_seconds(self, start, end):
        #only for original file that has datetimeindex
        time_1 = timedelta(seconds = start)
        time_2 = timedelta(seconds = end)
        
        time_start = pd.to_datetime(self.df.index[0], infer_datetime_format=True)
        time_1 = time_start + time_1 
        time_2 = time_start + time_2
        df2 = self.df.between_time(time_1.time(), time_2.time())

        self.df2 = df2
        #return df2
    

    def plot_burst(self):
        x = [x/(128*3600) for x in range(len(self.df.index))] #128 vectors a second
        fig = plt.figure()
        plt.subplot(4,1,1)
        plt.plot(x, self.df['IBS_X'], label = 'IBS')
        plt.plot(x, self.df['OBS_X'], 'r', label = 'OBS')
        plt.legend(loc='upper right')
        plt.ylabel('Bx [nT]')
        
        plt.subplot(4,1,2)
        plt.plot(x, self.df['IBS_Y'], label = 'IBS')
        plt.plot(x, self.df['OBS_Y'], 'r',label = 'OBS')
        plt.legend(loc='upper right')
        plt.ylabel('By [nT]')
        
        plt.subplot(4,1,3)
        plt.plot(x, self.df['IBS_Z'], label = 'IBS')
        plt.plot(x, self.df['OBS_Z'], 'r', label = 'OBS')
        plt.ylabel('Bz [nT]')
        plt.legend(loc='upper right')

        plt.subplot(4,1,4)
        plt.plot(x, self.df['IBS_MAGNITUDE'], label = 'IBS')
        plt.plot(x, self.df['OBS_MAGNITUDE'], 'r', label = 'OBS')
        plt.ylabel('B [nT]')
        plt.xlabel('Time [Hours]')
        plt.legend(loc='upper right')
        
        plt.suptitle('Magnetic Field with means removed')
        plt.show()


    def burst_powerspectra(self, OBS, *, df2 = False , name = ''):
        if OBS:
            collist = ['Time', 'OBS_X', 'OBS_Y', 'OBS_Z']
            name_str = 'OBS_burst'
        else:
            collist = ['Time', 'IBS_X', 'IBS_Y', 'IBS_Z']
            name_str = 'IBS_burst'
        if df2:
            df = self.df2
        else:
            df = self.df

        processing.powerspecplot(df, 128, collist, False, probe = 'MAG', inst = name_str, inflight = True, scaling = 'spectrum', name = name)

        
    def power_proportionality(self):
        x = self.df['OBS_MAGNITUDE'] 
        f_obs, Pxx_obs = sps.periodogram(x, self.fs, scaling='spectrum')
        y = self.df['IBS_MAGNITUDE'] 
        f_ibs, Pxx_ibs = sps.periodogram(y, self.fs, scaling='spectrum')
        
        division = Pxx_ibs/Pxx_obs
        division = division[division < 1000]
        
        print(division)
        print(np.median(division))
        n, bins, patches = plt.hist(division.flatten(), bins = 10000)
        plt.show()
        print(n)
        print(bins)


    def spectrogram(self, OBS, *, downlimit = 0, uplimit=0.001):
        if OBS:
            collist = ['Time', 'OBS_X', 'OBS_Y', 'OBS_Z']
            name_str = 'OBS_burst'
        else:
            collist = ['Time', 'IBS_X', 'IBS_Y', 'IBS_Z']
            name_str = 'IBS_burst'
            #x = np.sqrt(df[self.collist[1]]**2 + self.df[self.collist[2]]**2 + self.df[self.collist[3]]**2)
        y = (self.df[collist[1]] + self.df[collist[2]] + self.df[collist[3]])
        dflen = len(self.df)
        div = (dflen)/20000
        #f, Pxx = sps.periodogram(x,fs)
        #div = 500
        nff = dflen//div
        wind = sps.hamming(int(dflen//div))
        f, t, Sxx = sps.spectrogram(y, self.fs, window=wind, noverlap = int(dflen//(2*div)), nfft = nff)#,nperseg=700)
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
        #plt.semilogy()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        
        plt.title(f'MAG {name_str} Spectrogram @ {self.fs}Hz')
        plt.ylim((0,10))
        #plt.clim()
        fig = plt.gcf()
        ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.5, 1]
        if OBS:
            ticks = [0, 10e-6,10e-5,10e-4,10e-3,10e-2]
        cbar = plt.colorbar(ticks = ticks)
        #cbar.ax.set_yticklabels(fontsize=8)
        cbar.set_label('Normalised Power Spectral Density of the Trace')#, rotation=270)
        plt.show()

        return t,f,Sxx

    def moving_powerfreq(self, OBS, len_of_sections = 600, desired_freqs = [8.0], *, scaling = 'spectrum'):
        if OBS:
            collist = ['OBS_X', 'OBS_Y', 'OBS_Z', 'OBS_MAGNITUDE']
            name_str = 'OBS_burst'
        else:
            collist = ['IBS_X', 'IBS_Y', 'IBS_Z', 'IBS_MAGNITUDE']
            name_str = 'IBS_burst'
        
        probe_x = collist[0]
        probe_y = collist[1]
        probe_z = collist[2]
        mag = collist[3]

        self.df = self.df[collist] #reducing ram size

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

        sections = len(self.df)//(128*len_of_sections)
        start = 0
        end = len_of_sections*128

        for i in range(sections):
            df_tmp = self.df.iloc[start:end,:]
            x = df_tmp[probe_x]#[:20000]
            f_x, Pxx_x = sps.periodogram(x, self.fs, scaling = f'{scaling}')
            x_y = df_tmp[probe_y]#[:20000]
            f_y, Pxx_y = sps.periodogram(x_y, self.fs, scaling = f'{scaling}')
            x_z = df_tmp[probe_z]#[:20000]
            f_z, Pxx_z = sps.periodogram(x_z, self.fs, scaling = f'{scaling}')
            x_t = x + x_y + x_z #trace
            f_t, Pxx_t = sps.periodogram(x_t, self.fs, scaling = f'{scaling}')
            x_m = df_tmp[mag]
            f_m, Pxx_m = sps.periodogram(x_m, self.fs, scaling = f'{scaling}')

            if i == 0:
                x_dict = get_powerspec_of_desired_freq(f_x, Pxx_x, desired_freqs)
                y_dict = get_powerspec_of_desired_freq(f_y, Pxx_y, desired_freqs)
                z_dict = get_powerspec_of_desired_freq(f_z, Pxx_z, desired_freqs)
                t_dict = get_powerspec_of_desired_freq(f_t, Pxx_t, desired_freqs)
                m_dict = get_powerspec_of_desired_freq(f_m, Pxx_m, desired_freqs)
                #print(type(x_dict))
            else:
                x_dict_tmp = get_powerspec_of_desired_freq(f_x, Pxx_x, desired_freqs)
                y_dict_tmp = get_powerspec_of_desired_freq(f_y, Pxx_y, desired_freqs)
                z_dict_tmp = get_powerspec_of_desired_freq(f_z, Pxx_z, desired_freqs)
                t_dict_tmp = get_powerspec_of_desired_freq(f_t, Pxx_t, desired_freqs)
                m_dict_tmp = get_powerspec_of_desired_freq(f_m, Pxx_m, desired_freqs)

                for j in desired_freqs:
                    x_dict[str(j)].append(x_dict_tmp[str(j)][0])
                    y_dict[str(j)].append(y_dict_tmp[str(j)][0])
                    z_dict[str(j)].append(z_dict_tmp[str(j)][0])
                    t_dict[str(j)].append(t_dict_tmp[str(j)][0])
                    m_dict[str(j)].append(m_dict_tmp[str(j)][0])
            
            start += len_of_sections
            end += len_of_sections

        #print(x_dict[str(8.0)])
        
        plt.figure()
        #plt.plot(range(sections), x_dict[str(8.0)], label = 'X')
        #plt.plot(range(sections), y_dict[str(8.0)], label = 'Y')
        #plt.plot(range(sections), z_dict[str(8.0)], label = 'Z')
        x = [i*len_of_sections/3600 for i in range(sections)]
        plt.plot(x, t_dict[str(8.0)], label = 'T - 8Hz')
        plt.plot(x, t_dict[str(16.0)], label = 'T - 16Hz')
        plt.plot(x, t_dict[str(0.119)], label = 'T - 0.119Hz')
        plt.plot(x, t_dict[str(0.238)], label = 'T - 0.238Hz')
        plt.plot(x, t_dict[str(0.357)], label = 'T - 0.357Hz')
        plt.plot(x, t_dict[str(0.596)], label = 'T - 0.596Hz')
        
        #plt.plot(range(sections), m_dict[str(8.0)], label = 'M')
        plt.legend(loc='upper right')
        plt.ylabel('Power [dB]')
        plt.xlabel('Time [Hours]')
        plt.title(f'{len_of_sections//60} min moving max Power')
        plt.semilogy()
        plt.show()
        

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
    
    burst_object = burst_data()
    burst_object.get_df_from_mat(file_one=False, start = int(128*3600*0.3), end = int(128*3600*24)) #0.3 to 24, 24 to 47.6 and 48.3 to 72
    #burst_object.plot_burst()
    OBS = True

    burst_object.moving_powerfreq(OBS,len_of_sections=300,desired_freqs=[0.119, 0.238, 0.596, 0.357, 8.0, 16.0])

    #burst_object.spectrogram(OBS, downlimit = 0, uplimit = 0.001) #0.005
    #burst_object.burst_powerspectra(OBS, name = '_file2_day2')

    #burst_object.df_to_csv(name='file_2_day_1')


    """
    burst_object = burst_data()
    burst_object.get_df_from_csv(day=1) #takes 64 seconds to read in day, reading mat is four times faster
    """

    #burst_object = burst_data(file_one=False, start = int(128*3600*0.1), end = int(128*3600*1))
    #burst_object = burst_data()
    #OBS = False

    #burst_object.spectrogram(OBS, downlimit = 0, uplimit = 0.005) #0.005

    
    #thruster at start and at 48 hours
    #burst_object.plot_burst()
    #burst_object.get_df_between_seconds(33000, 33400)
    #w0 = 8/(128/2)
    #b,a = sps.iirnotch(w0, Q=30)
    #burst_object.burst_powerspectra(OBS, df2 = True)

    """
    t,f,Sxx_ibs = spectrogram(df, OBS, downlimit = 0, uplimit=0.001)
    t,f,Sxx_obs = spectrogram(df, True, downlimit = 0, uplimit=0.001)
    Sxx_dif = Sxx_ibs - Sxx_obs
    plt.pcolormesh(t, f, Sxx_dif, vmin = 0, vmax = 0.001, cmap = 'viridis') #sqrt? 
    #plt.pcolormesh(t, f, Sxx, clim = (0,uplimit))
    plt.semilogy()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    
    plt.title(f'IBS-OBS Sxx Spectrogram @ 128Hz')
    plt.ylim((10**0,128/2))
    #plt.clim()
    fig = plt.gcf()
    ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.5, 1]
    if OBS:
        ticks = [0, 10e-6,10e-5,10e-4,10e-3,10e-2]
    cbar = plt.colorbar(ticks = ticks)
    #cbar.ax.set_yticklabels(fontsize=8)
    cbar.set_label('Normalised Power Spectral Density of the Trace')#, rotation=270)
    plt.show()
    """
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