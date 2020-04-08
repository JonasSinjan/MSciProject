import pandas as pd
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scipy.signal as sps
import scipy.stats as spstats
from datetime import datetime, timedelta
import time
import math
from tqdm import tqdm
import csv

class processing:
    
    #@profile
    @staticmethod
    def read_files(all_files, soloA, sampling_freq = None, collist=None, day=1, start_dt = None, end_dt = None): #removed windows after soloA for mfsa object and also redundant
        #path - location of folder to concat
        #soloA - set to True if soloA, if soloB False
        li = [] 
        for filename in tqdm(all_files):   
            if soloA:
                if collist == None:
                    df =  pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';', dtype = np.float32)
                    cols = df.columns.tolist()
                    new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #this will reorder the columns into the correct order
                    df = df[new_cols]
                else:
                    df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';', usecols = collist, dtype = np.float32)#header = 350, nrows = rows)
            else:
                if collist == None:
                    df =  pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';', dtype = np.float32)
                    cols = df.columns.tolist()
                    new_cols = [cols[0]] + cols[9:13] + cols[1:9] + cols[13:17]
                    df = df[new_cols]
                else:
                    df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';', usecols = collist, dtype = np.float32)#, header = 170, nrows = rows)
                
            li.append(df)
        #tqdm.pandas(desc="Progress Bar")
        df = pd.concat(li, ignore_index = True, sort=True)

        """
        #factor = int(1000/freq_max)
        if sampling_freq != None:
            factor = int(1000/sampling_freq)
            assert type(factor) == int
            print(factor)
            df = df.groupby(np.arange(len(df))//factor).mean()
        """    
        df = df.sort_values('time', ascending = True, kind = 'mergesort')

        if soloA:
            if '21' in all_files[0]: #for day_one
                start_second = df['time'][0] + 10.12
                start_dt_time = pd.to_datetime(start_second, unit = 's', origin = '2019-06-24 08:10:00' )

                #df['time'] = df['time'] + 10.12
                #df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-21 08:10:00' )
            elif '24' in all_files[0]: #for day_two
                start_second = df['time'][0] + 46.93
                start_dt_time = pd.to_datetime(start_second, unit = 's', origin = '2019-06-24 08:14:00' )
                
                #df['time'] = df['time'] + 46.93
                #df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-24 08:14:00' )
        else:
            if '21' in all_files[0]:
                start_second = df['time'][0] + 10
                start_dt_time = pd.to_datetime(start_second, unit = 's', origin = '2019-06-21 08:09:00' )

                #df['time'] = df['time'] + 10
                #df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-21 08:09:00' )
            elif '24' in all_files[0]:
                start_second = df['time'][0] + 24
                start_dt_time = pd.to_datetime(start_second, unit = 's', origin = '2019-06-24 08:14:00' )

                #df['time'] = df['time'] + 24
                #df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-24 08:14:00' )

        seconds = len(df)//1000
        microseconds = (len(df)/1000 - seconds)*1000000
        end_time = start_dt_time + timedelta(seconds = seconds, microseconds=microseconds)
        date_range = pd.date_range(start = start_dt_time, end = end_time, freq='1000000ns') #1/128 seconds exactly for 1/16 just need microseconds 'ms'
        print(len(date_range), len(df))
        df['time'] = date_range[:-1]

        df['time'] = df['time'].dt.round('ms')
        #df = df.sort_values('time', ascending = True, kind = 'mergesort')
        df.set_index('time', inplace = True)

        #print(df.head())
        #print(df.tail())

        if sampling_freq < 1000:
            factor = int(1000/sampling_freq)
            if factor >= 0.001:
                df = df.resample(f'{factor}ms').mean()
            else:
                print('The resampling is in the wrong units - must be factor*milliseconds')
        else:
            print('The desired sampling frequency is greater than the raw data available - defaulted to 1kHz')

        return df

    @staticmethod
    def which_csvs(soloA_bool, day, start_dt, end_dt, tz_MAG = False):
        if tz_MAG:
            day_one_A_dt = datetime(2019,6,21,8,10,10,12) - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283) 
            day_one_B_dt = datetime(2019,6,21,8,9,10) - pd.Timedelta(days = 0, hours = 1, minutes = 58, seconds = 46, milliseconds = 499)
            day_two_A_dt = datetime(2019,6,24,8,14,46,93) - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283) 
            day_two_B_dt = datetime(2019,6,24,8,14,24) - pd.Timedelta(days = 0, hours = 1, minutes = 58, seconds = 46, milliseconds = 499)
        else:
            day_one_A_dt = datetime(2019,6,21,8,10,10,12)
            day_one_B_dt = datetime(2019,6,21,8,9,10)
            day_two_A_dt = datetime(2019,6,24,8,14,46,93)
            day_two_B_dt = datetime(2019,6,24,8,14,24)

        length = (end_dt - start_dt).total_seconds()
        #print(length)
        
        if soloA_bool:
            if day == 1 or day == 21:
                time_delta = (start_dt - day_one_A_dt).total_seconds()
            else:
                time_delta = (start_dt - day_two_A_dt).total_seconds()
            start_csv = math.floor(time_delta / 384) # approx number of csv files
            end_csv = start_csv + math.ceil(length/384) + 3
            if day == 1:
                if end_csv > 81:
                    end_csv = 81
                    print('The desired time range may run outside the available data - check if so')
            else:
                if end_csv > 83:
                    end_csv = 83
                    print('The desired time range may run outside the available data - check if so')
            #print(length/384, math.ceil(length/384))
        else:
            if day == 1 or day == 21:
                time_delta = (start_dt - day_one_B_dt).total_seconds()
            else:
                time_delta = (start_dt - day_two_B_dt).total_seconds()
            start_csv = math.floor(time_delta / 658) # approx number of csv files
            end_csv = start_csv + math.ceil(length/658)
            if day == 1:
                if end_csv > 47:
                    end_csv = 47
                    print('The desired time range may run outside the available data - check if so')
            else:
                if end_csv > 48:
                    end_csv = 48
                    print('The desired time range may run outside the available data - check if so')
        
        #if start_csv == 0:
        #    start_csv = 1
        return start_csv, end_csv

    @staticmethod
    def shifttime(df, soloAbool, day):
        if soloAbool:
            if day == 1:
                df.index = df.index - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 30, milliseconds = 137)
            else:
                df.index = df.index - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)  
        else:
            if day == 1:
                df.index = df.index - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 1, milliseconds = 606)
            else:
                df.index = df.index - pd.Timedelta(days = 0, hours = 1, minutes = 58, seconds = 46, milliseconds = 499)  
            
        return df

    @staticmethod
    def calculate_dB(df, peak_datetimes):
        step_dict = {}
        time_to_avg = 30 #need the 2 seconds for buffer time, as exact timestamp of current has uncertainty of ~2 seconds either side (current at 5sec resample)
        buffer = 2
        time_to_avg += buffer
        for k in df.columns.tolist(): #looping through x, y, z
            print(k)
            if str(k) not in step_dict.keys():
                step_dict[str(k)] = 0
                
            tmp_step_list = [0]*len(peak_datetimes)
            tmp_step_err_list = [0]*len(peak_datetimes)
            #print(len(peak_datetimes))
            for l, time in enumerate(peak_datetimes): #looping through the peaks datetimes
                
                if l == 0:
                    time_before_left = time - pd.Timedelta(seconds = time_to_avg)
                else:
                    #time_before_left = peak_datetimes[l-1] + pd.Timedelta(seconds = 2) #old method to average over maximum possible time
                    tmp = time - pd.Timedelta(seconds = time_to_avg)
                    if tmp > peak_datetimes[l-1] + pd.Timedelta(seconds = buffer): #checking to see which is later, if time distance between two peaks less than a minute
                        time_before_left = tmp
                    else:
                        time_before_left = peak_datetimes[l-1] + pd.Timedelta(seconds = buffer)
                    
                time_before_right = time - pd.Timedelta(seconds = buffer) #buffer time since sampling at 5sec, must be integers
                time_after_left = time + pd.Timedelta(seconds = buffer)
                
                if l == len(peak_datetimes)-1:
                    time_after_right = time + pd.Timedelta(seconds = time_to_avg)
                else:
                    #time_after_right = peak_datetimes[l+1] - pd.Timedelta(seconds = 2) # old method to average over maximum possible time
                    tmp = time + pd.Timedelta(seconds = time_to_avg)
                    if tmp < peak_datetimes[l+1] - pd.Timedelta(seconds = buffer):
                        time_after_right = tmp
                    else:
                        time_after_right = peak_datetimes[l+1] - pd.Timedelta(seconds = buffer)
                    
                df_tmp = df[str(k)]
                df_before = df_tmp.between_time(time_before_left.time(), time_before_right.time())
                avg_tmp = df_before.mean()
                std_before = df_before.std()/np.sqrt(len(df_before))
                
                df_after = df_tmp.between_time(time_after_left.time(), time_after_right.time())
                avg_after_tmp = df_after.mean()
                std_after = df_after.std()/np.sqrt(len(df_after))


                step_tmp = avg_after_tmp - avg_tmp
                step_tmp_err = np.sqrt(std_before**2 + std_after**2)

                #print(step_tmp,step_tmp_err)
                
                if math.isnan(step_tmp):
                    print(l, time)
                    print(time_before_left, time_before_right)
                    print(time_after_left, time_after_right)
                
                tmp_step_list[l] = step_tmp
                tmp_step_err_list[l] = step_tmp_err
                
            step_dict[str(k)] = tmp_step_list
            step_dict[str(k) + ' err'] = tmp_step_err_list

        return step_dict

    @staticmethod
    def powerspecplot(df, fs, collist, alt, inst = None, save = False, *, probe=None, inflight = False, scaling = 'density', name='', ten_milly = False):
        start = time.time()
        clicks = []
        def onclick(event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                ('double' if event.dblclick else 'single', event.button, event.x, event.y, event.xdata, event.ydata))
            clicks.append(event.xdata)
            clicks.append(event.ydata)

        probe_x = collist[1]
        probe_y = collist[2]
        probe_z = collist[3]
        #probe_m = collist[4]
        x = df[probe_x]#[:20000]
         #nfft = 10_000_000
        x_y = df[probe_y]#[:20000]
         #nfft = 10_000_000,
        x_z = df[probe_z]#[:20000]
         #nfft = 10_000_000
        #x = df[probe_m]#[:20000]
        #f_m, Pxx_m = sps.periodogram(x,fs, scaling='spectrum')
        x_t = x + x_y + x_z #trace
        if ten_milly:
            f_x, Pxx_x = sps.periodogram(x, fs, nfft = 10_000_000, scaling=f'{scaling}')
            f_y, Pxx_y = sps.periodogram(x_y, fs, nfft = 10_000_000, scaling=f'{scaling}')
            f_z, Pxx_z = sps.periodogram(x_z, fs, nfft = 10_000_000, scaling=f'{scaling}')
            f_t, Pxx_t = sps.periodogram(x_t, fs, nfft = 10_000_000, scaling =f'{scaling}') #nfft = 10_000_000,
        else:
            f_x, Pxx_x = sps.periodogram(x, fs, scaling=f'{scaling}')
            f_y, Pxx_y = sps.periodogram(x_y, fs, scaling=f'{scaling}')
            f_z, Pxx_z = sps.periodogram(x_z, fs, scaling=f'{scaling}')
            f_t, Pxx_t = sps.periodogram(x_t, fs, scaling =f'{scaling}')

        def filter_Pxx(f,Pxx, mask_frequencies, harmonics):
            harmonics = range(3,400,2)#[3,5,7,8,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41]
            dfreq = 0.02
            #index = []

            for i in range(len(harmonics)):
                index_tmp = np.where((f >= mask_frequencies*harmonics[i] - dfreq/2 ) & (f <= mask_frequencies*harmonics[i] + dfreq/2))
                Pxx[index_tmp] = 0

            #Pxx[index] = 0

            return Pxx
        
        def plot_power(f,fs,Pxx, probe, col):
            #Pxx = filter_Pxx(f, Pxx, 0.119, 2)
            plt.loglog(f,np.sqrt(Pxx), f'{col}-', picker=100) #sqrt required for power spectrum, and semi log y axis
            plt.xlim(left = 1e-4, right=fs/2)
            if inflight:
                plt.ylim(bottom = 10e-4, top = 10e4)
            else:
                plt.ylim(bottom = 10e-2, top = 10e1)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude Power Spectral Density [nT$/\sqrt{Hz}$]')
            if probe.split('_')[0] == 'Probe09':
                probe = 'Probe10_' + str(probe.split('_')[1])
            elif probe.split('_')[0] == 'Probe10':
                probe = 'Probe09_' + str(probe.split('_')[1])
            plt.title(f'{probe}')
            
            peaks, _ = sps.find_peaks(np.log10(Pxx), prominence = 6)
            #print(peaks)
            
            #peaks = peaks[np.where(f[peaks] > 0)]
            print(probe, [round(i,1) for i in f[peaks] if i <= fs/2], len(peaks))
            
        def get_clicks(f, Pxx, Probe):
            fig = plt.figure()
            plot_power(f, fs, Pxx, Probe, 'b')
            plt.xlim(left = 10e-2)
            plt.ylim(top = 10e1)
            mpl.rcParams['agg.path.chunksize'] = 10000
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

            return clicks
        if probe == None:
                probe = probe_x.split('_')[0]
                print(probe, inst)
        try:
            print(f'Trying to find file: {probe}_{inst}_powerspectra{name}.csv')
            with open(f'.\\Results\\PowerSpectrum\\Peak_files\\{probe}_{inst}_powerspectra{name}.csv') as f:
                clicks1, clicks2, clicks3, clicks4 = [], [], [], []
                for i, line in enumerate(f):
                    nums = line.split(',')
                    nums1 = nums[1]
                    nums2 = nums[2]
                    if line[0] == 'X':
                        clicks1.append(float(nums1))
                        clicks1.append(float(nums2))
                    if line[0] == 'Y':
                        clicks2.append(float(nums1))
                        clicks2.append(float(nums2))
                    if line[0] == 'Z':
                        clicks3.append(float(nums1))
                        clicks3.append(float(nums2))
                    if line[0] == 'T':
                        clicks4.append(float(nums1))
                        clicks4.append(float(nums2))

        except IOError:
            print("File does not exist - Now will be created")    
            
            clicks1 = get_clicks(f_x, Pxx_x, probe_x)
            clicks = []

            clicks2 = get_clicks(f_y, Pxx_y, probe_y)
            clicks = []

            clicks3 = get_clicks(f_z, Pxx_z, probe_z)
            clicks = []

            clicks4 = get_clicks(f_t, Pxx_t, 'T')
            clicks = []

            print(clicks1, type(clicks1))
            print(clicks2, type(clicks2))
            print(clicks3, type(clicks3))
            print(clicks4, type(clicks4))

            def write_peaks(clicks, dir):
                clicks = [round(j,4) for j in clicks]
                i = 0
                for k in range(int(len(clicks)/2)):
                    w.writerow([dir, clicks[i], clicks[i+1]])
                    i += 2

            #probe = probe_x.split('_')[0]
            w = csv.writer(open(f".\\Results\\PowerSpectrum\\Peak_files\\{probe}_{inst}_powerspectra{name}.csv", "w", newline=''))
            #w.writerow(["Probe","X.slope_lin", "Y.slope_lin", "Z.slope_lin","X.slope_lin_err", "Y.slope_lin_err", "Z.slope_lin_err","X_zero_err","Y_zero_err","Z_zero_err"])#,"X.slope_curve", "Y.slope_curve", "Z.slope_curve","X.slope_curve_err", "Y.slope_curve_err", "Z.slope_curve_err"])
            w.writerow(["Dir", "Xdata", "Ydata"])
            write_peaks(clicks1, "X")
            write_peaks(clicks2, "Y")
            write_peaks(clicks3, "Z")
            write_peaks(clicks4, "T")

        def plot_peaks(clicks, axis):
            j = 0
            peaks = clicks
            for i in range(int(len(peaks)/2)):
                axis.loglog(peaks[j], peaks[j+1], marker = 's', markersize = 5, color='orange', linestyle = 'None', markeredgecolor='black')
                #axis.annotate(f'{round(peaks[j],2)}', (peaks[j], peaks[j+1]), xytext = (peaks[j] - 10**0.6, peaks[j+1] + 1), wrap = True)
                j += 2

        fig = plt.figure(figsize = (10,8))#, ax = plt.subplots(2, 2, figsize = (10,8))
        mpl.rcParams['agg.path.chunksize'] = 10000
        uplim = 10e0 #11 otherwise, 50 only for probe 12
        if probe_x == 'Probe12_X':
            uplim = 50
        elif inflight == True:
            downlim = 10e-4
        else:
            downlim = 10e-3
        """
        elif scaling == 'spectrum':
            downlim = 10e-7
        """
        uplim = 10e1
        downlim = 10e-4
        ax1 = plt.subplot(221)
        plot_power(f_x, fs, Pxx_x, probe_x, 'b')
        plt.ylim(downlim, uplim)
        #plt.xlim(left = 0.5*10e0)
        plot_peaks(clicks1, ax1)
        
        
        ax2 = plt.subplot(222)
        plot_power(f_y, fs, Pxx_y, probe_y, 'r')
        plt.ylim(downlim, uplim)
        #plt.xlim(left = 0.5*10e0)
        plot_peaks(clicks2, ax2)
        
        ax3 = plt.subplot(223)
        plot_power(f_z, fs, Pxx_z, probe_z, 'g')
        plt.ylim(downlim, uplim)
        #plt.xlim(left = 0.5*10e0)
        plot_peaks(clicks3, ax3)
        
        ax4 = plt.subplot(224)
        Trace = 'Trace'
        plot_power(f_t, fs, Pxx_t, Trace, 'y')
        plt.ylim(downlim, uplim)
        #plt.xlim(left = 0.5*10e0)
        plot_peaks(clicks4, ax4)
       
        plt.suptitle(f'{inst} - Power Spectrum')
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        plt.show()
        

        def alt_power_spec(data, fs, probe):
            ps = np.abs(np.fft.fft(data))**2
            time_step = 1/fs
            freqs = np.fft.fftfreq(len(data), time_step)
            idx = np.argsort(freqs)
            plt.loglog(freqs[idx], ps[idx])
            plt.xlim(right=fs/2)
            plt.title(f'{probe}')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('abs(FFT(data)**2)')
            #plt.ylim(10e1,10e5)
            
        if alt:
            plt.figure()
            plt.subplot(221)
            alt_power_spec(x, fs, probe_x)

            plt.subplot(222)
            alt_power_spec(x_y, fs, probe_y)

            plt.subplot(223)
            alt_power_spec(x_z, fs, probe_z)

            probe_t = 'Trace'
            plt.subplot(224)
            alt_power_spec(x_t, fs, probe_t)

        if save:
            plt.savefig(f'.\\Results\\PowerSpectrum\\Day_2\\{probe}_{inst}_powerspec.png')
        # else:
        #     plt.show()
        print('Power Spectrum successfully completed\nExecution time: ', round(time.time() - start,3), ' seconds')

    @staticmethod
    def rotate_21(soloA_bool):
        if soloA_bool:
            M_1 = np.array([-0.01151451,-0.03379413,-0.99912329,-0.99966879,0.02182397,0.01062976,0.02152466,0.99893068,-0.03362521])
            M_2 = np.array([0.0221395,-0.00814719,-1.00104386,-0.99984468,0.04330073,-0.02333449,0.04405573,1.00084827,-0.00708282])
            M_3 = np.array([0.03283952,-0.00829181,-0.99809581,-0.99786078,-0.01761458,-0.03303076,-0.01707146,0.99865858,-0.00872099])
            M_4 = np.array([0.00742014,-0.00079088,-1.00090218,-1.00039189,0.02286711,-0.00760828,0.02237911,1.00026345,-0.00006682])
            M_5 = np.array([-0.03153728,0.01160465,1.00256374,0.99654512,0.10814223,0.03088474,-0.10843654,0.99619989,-0.01422552])
            M_6 = np.array([-0.00294161,-0.01878043,-0.99875921,-0.99913433,-0.00545105,0.00357126,-0.004815,0.99911992,-0.01849102])
            M_7 = np.array([-0.01694624,0.00893438,-1.00254205,-1.00234939,-0.00859451,0.01676781,-0.00850761,1.0027674,0.00960575])
            M_8 = np.array([-0.01233755,-0.00211036,-1.00689605,-1.00708674,-0.02531075,0.01233301,-0.02576082,1.00706965,-0.00212613])

            M_A = [M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8]
            
            for i, M in enumerate(M_A):
                M_A[i] = M.reshape((3, 3))
                
            return M_A
                
        else:
            M_10 = np.array([0.00529863,-0.00657411,-0.99965402,0.92140194,-0.38605527,0.00704625,-0.38633163,-0.92200509,0.0042057])
            M_9 = np.array([-0.05060716,-1.00568091,-0.03885759,-1.00772239,0.05060271,0.0010825,0.0006611,0.04126364,-1.0028714])
            M_11 = np.array([0.09562948,-0.99808126,-0.00550316,-0.0083807,0.00490206,-1.00708301,0.99761069,0.10039216,-0.00864757])
            M_12 = np.array([-5.40867212,1.13925683,-3.46278696,-0.72430491,0.7475252,4.3949523,1.28427441,7.06375231,-0.4813982])

            M_B = [M_9,M_10,M_11,M_12]
            
            for i, M in enumerate(M_B):
                M_B[i] = M.reshape((3, 3))
                
            return M_B
        
    @staticmethod
    def rotate_24(soloA_bool):
        if soloA_bool:
            M_1 = np.array([-0.01151451,-0.03379413,-0.99912329,-0.99966879,0.02182397,0.01062976,0.02152466,0.99893068,-0.03362521])
            M_2 = np.array([0.0221395,-0.00814719,-1.00104386,-0.99984468,0.04330073,-0.02333449,0.04405573,1.00084827,-0.00708282])
            M_3 = np.array([0.03283952,-0.00829181,-0.99809581,-0.99786078,-0.01761458,-0.03303076,-0.01707146,0.99865858,-0.00872099])
            M_4 = np.array([0.00742014,-0.00079088,-1.00090218,-1.00039189,0.02286711,-0.00760828,0.02237911,1.00026345,-0.00006682])
            M_5 = np.array([-0.03153728,0.01160465,1.00256374,0.99654512,0.10814223,0.03088474,-0.10843654,0.99619989,-0.01422552])
            M_6 = np.array([-0.00294161,-0.01878043,-0.99875921,-0.99913433,-0.00545105,0.00357126,-0.004815,0.99911992,-0.01849102])
            M_7 = np.array([-0.01694624,0.00893438,-1.00254205,-1.00234939,-0.00859451,0.01676781,-0.00850761,1.0027674,0.00960575])
            M_8 = np.array([-0.01233755,-0.00211036,-1.00689605,-1.00708674,-0.02531075,0.01233301,-0.02576082,1.00706965,-0.00212613])

            M_A = [M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8]
            
            for i, M in enumerate(M_A):
                M_A[i] = M.reshape((3, 3))
                
            return M_A
                
        else:
            M_10 = np.array([0.00529863,-0.00657411,-0.99965402,0.92140194,-0.38605527,0.00704625,-0.38633163,-0.92200509,0.0042057])
            M_9 = np.array([-0.05060716,-1.00568091,-0.03885759,-1.00772239,0.05060271,0.0010825,0.0006611,0.04126364,-1.0028714])
            M_11 = np.array([0.09562948,-0.99808126,-0.00550316,-0.0083807,0.00490206,-1.00708301,0.99761069,0.10039216,-0.00864757])
            M_12 = np.array([-5.40867212,1.13925683,-3.46278696,-0.72430491,0.7475252,4.3949523,1.28427441,7.06375231,-0.4813982])

            M_B = [M_9,M_10,M_11,M_12]
            
            for i, M in enumerate(M_B):
                M_B[i] = M.reshape((3, 3))
                
            return M_B
                
    #rotate_21(True)
    @staticmethod
    def soloA(file_path):
        #skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
        df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';')
        cols = df.columns.tolist()
        new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #reorder the columns into the correct order
        df = df[new_cols]
        return df

    @staticmethod
    def soloB(file_path):
        #skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
        df_B = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';')
        cols = df_B.columns.tolist()
        new_cols = [cols[0]] + cols[9:13] + cols[1:9] + cols[13:17]#reorder the columns into the correct order # adding time as first column
        df_B = df_B[new_cols]
        return df_B



"""
 all_folders = glob.glob(path_fol_A + "\*")
        #print(all_folders)
        li, length = [], []
        for folder in all_folders:
            all_files = glob.glob(folder + "\*.csv")
            for filename in all_files:
                df =  pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';', usecols = ['time'])
                length.append(len(df))
                li.append(df)
            df = pd.concat(li, ignore_index = True, sort=True)
            print(folder, ', seconds = ', len(df)/1000, ', mins = ',len(df)/60000, ', hours = ', len(df)/3600000)
            li = []
"""



