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


def day_two(windows, probe_num_list, start_dt, end_dt, alt = False, sampling_freq = None, plot = True, spectrogram = False, powerspec = False, inst_name=None):
    if windows:
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_two\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_two\B'
    else:
        path_fol_A =  os.path.expanduser("~/Documents/MsciProject/Data/day_two/A")
        path_fol_B =  os.path.expanduser("~/Documents/MsciProject/Data/day_two/B")
    
    #here select which probe is desired, only one at a time
    #num = 12
    b_noise = []
    for num in probe_num_list:
        print('num = ', num)
        if num < 9:
            soloA_bool = True
        else:
            soloA_bool = False
        if num <10:
            num_str = f'0{num}'
        else: 
            num_str = num
        collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']

        #finding the correct MFSA data files
        start_csv, end_csv = processing.which_csvs(soloA_bool, 2, start_dt, end_dt, tz_MAG=False) #this function (in processing.py) finds the number at the end of the csv files we want
        print(start_csv, end_csv)

        all_files = [0]*(end_csv + 1 - start_csv)
        
        for index, i in enumerate(range(start_csv, end_csv + 1)): #this will loop through and add the csv files that contain the start and end time set above
            if soloA_bool:
                if windows:
                    all_files[index] = path_fol_A + f'\SoloA_2019-06-24--08-14-46_{i}.csv'
                else:
                    all_files[index] = path_fol_A + os.path.expanduser(f'/SoloA_2019-06-24--08-14-46_{i}.csv') #need to change path_fol_A  to the path where your A folder is
            else:
                if windows:
                    all_files[index] = path_fol_B + f'\SoloB_2019-06-24--08-14-24_{i}.csv'
                else:
                    all_files[index] = path_fol_B + os.path.expanduser(f'/SoloB_2019-06-24--08-14-24_{i}.csv') #need to change path_fol_B to the path where your B folder is
    #set this to the directory where the data is kept on your local computer
        if soloA_bool:
            df = processing.read_files(all_files, soloA_bool, windows, sampling_freq, collist, day=2, start_dt = start_dt, end_dt = end_dt)
            rotate_mat = processing.rotate_24(soloA_bool)[num-1]
        else:
            df = processing.read_files(all_files, soloA_bool, windows, sampling_freq, collist, day=2, start_dt = start_dt, end_dt = end_dt)
            rotate_mat = processing.rotate_24(soloA_bool)[num-9]
        df.iloc[:,0:3] = np.matmul(rotate_mat, df.iloc[:,0:3].values.T).T
        
        #find the df of the exact time span desired
        df2 = df.between_time(start_dt.time(), end_dt.time()) 
        dflen = len(df2)
        #df3 = df2.resample('1s').mean()
        
        #shitfing time so the axes are in spacecraft time to compare with current data
        df2 = processing.shifttime(df2, soloA_bool, 2)
        
        
        if plot: #plotting the raw probes results
            #plt.figure()
            df3 = df2.resample('1s').mean()
            tmp = []
            for col in collist[1:]:
                df2[col] = df2[col] - df2[col].mean()
                plt.plot(df2.index.time, df2[col], label=str(col))
                
                var_1hz = np.std(df3[col])
                var_1khz = np.std(df2[col])
                print('std - 1Hz', col, var_1hz)
                print('std - 1kHz', col,  var_1khz)
                tmp.append(var_1hz)
                tmp.append(var_1khz)
                #print(df2[col].abs().idxmax())
            
            plt.xlabel('Time (s)')
            plt.ylabel('dB (nT)')
            plt.title(f'Probe {num} @ {sampling_freq}Hz, {start_dt.date()}')
            plt.legend(loc="best")
            plt.show()

            return tmp

        if spectrogram:
            x = np.sqrt(df2[collist[1]]**2 + df2[collist[2]]**2 + df2[collist[3]]**2)
            fs = sampling_freq
            div = dflen/1000
            #f, Pxx = sps.periodogram(x,fs)
            #div = 500
            nff = dflen//div
            wind = sps.hamming(int(dflen//div))
            f, t, Sxx = sps.spectrogram(x,fs, window=wind, noverlap = int(dflen//(2*div)), nfft = nff)#,nperseg=700)
            ax = plt.figure()
            plt.pcolormesh(t, f, Sxx, vmin = 0.,vmax = 0.03)
            plt.semilogy()
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title(f'Spectrogram: Probe {num} @ {sampling_freq}Hz, {start_dt.date()}')
            plt.ylim((10**0,sampling_freq/2))
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
            
        if powerspec:
            if inst_name == None:
                processing.powerspecplot(df2, sampling_freq, collist, alt, save = False)
            else:
                processing.powerspecplot(df2, sampling_freq, collist, alt, inst = inst_name, save = True)

if __name__ == "__main__":
    
    windows = True
    probe_num_list = [7] #['STIX', 'METIS', 'SPICE', 'PHI', 'SoloHI', 'EUI', 'SWA', 'EPD']
    # METIS - 10:10-10:56
    # EUI - 9:24-10:09
    # SPICE - 10:57-11:18
    # STIX - 11:44-12:17
    # SWA - 12:18-13:52
    # PHI - 8:05-8:40
    # SoloHI - 11:19-11;44
    # EPD - 14:43-14:59 #be wary as epd in different regions #full ==>13:44-14:58

    #the datetime we change here is in spacecraft time - used for if want probes for a certain current profile (which is in spacecraft time)
    start_dt = datetime(2019,6,24,8,20) + pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
    end_dt = datetime(2019,6,24,15,00) + pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)# this is the end
    #start and end dt now in MFSA (German UT) time - as MFSA in that time
        
    #alt = False #if want powerspec from `brute force' method - or inbuilt scipy periodogram method
    tmp = day_two(windows, probe_num_list, start_dt, end_dt, sampling_freq = 100, plot = False, spectrogram = True, powerspec = False) #pass through the list containing the file paths
    
    """
    inst_hour_times = [10,10,9,10,10,11,11,12,12,13,8,8,11,11,14,14]
    inst_min_times = [10,56,24,9,57,18,44,17,18,52,4,40,19,44,43,59]
    inst_names = ['METIS', 'EUI', 'SPICE', 'STIX', 'SWA', 'PHI', 'SoloHI', 'EPD']
    i = 0
    for k in inst_names:
        start_dt = datetime(2019,6,24,inst_hour_times[i],inst_min_times[i]) + pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
        end_dt = datetime(2019,6,24,inst_hour_times[i+1],inst_min_times[i+1]) + pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283)# this is the end
        i = i + 2
        tmp = day_two(windows, probe_num_list, start_dt, end_dt, sampling_freq = 100, plot = False, spectrogram = False, powerspec = True, inst_name = k)

    """
    """
    b_noise.extend(tmp)
    w = csv.writer(open(f"day2_mfsa_probe_vars.csv", "w"))
    w.writerow(["Probe","Bx_var","By_var","Bz_var","Bx_var_1k","By_var_1k","Bz_var_1k"])
    val = b_noise
    j = 0
    for i in range(12):
        w.writerow([i+1,val[j],val[j+2],val[j+4],val[j+1],val[j+3],val[j+5]])#,val[9],val[10],val[11]])
        j += 6
    """