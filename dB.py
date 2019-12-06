import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import *
from pandas.plotting import register_matplotlib_converters
from current import current_peaks
register_matplotlib_converters()
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time
from datetime import datetime
import scipy.stats as spstats
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def dB(peak_datetimes, instrument, current_dif, jonas, probe_list, plot=False): #for only one instrument

    if jonas:
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_two\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_two\B'
    else:
        path_fol_A = os.path.expanduser("~/Documents/MSciProject/Data/day_two/A")
        path_fol_B = os.path.expanduser("~/Documents/MSciProject/Data/day_two/B")
    
    start_dt = peak_datetimes[0]-pd.Timedelta(minutes = 1)
    end_dt = peak_datetimes[-1]+pd.Timedelta(minutes = 1)
    
    day = 2 #second day
    sampling_freq = 1 #do we want to remove the high freq noise?
    
    start_csv_A, end_csv_A = which_csvs(True, day ,start_dt, end_dt, tz_MAG = True)
    start_csv_B, end_csv_B = which_csvs(False, day ,start_dt, end_dt, tz_MAG = True)

    all_files_A = [0]*(end_csv_A + 1 - start_csv_A)
    for index, j in enumerate(range(start_csv_A, end_csv_A + 1)): #this will loop through and add the csv files that contain the start and end time set above
        if jonas:
            all_files_A[index] = path_fol_A + f'\SoloA_2019-06-24--08-14-46_{j}.csv'
        else:
            all_files_A[index] = path_fol_A + os.path.expanduser(f'/SoloA_2019-06-24--08-14-46_{j}.csv') #need to change path_fol_A  to the path where your A folder is
    
    all_files_B = [0]*(end_csv_B + 1 - start_csv_B)
    for index, j in enumerate(range(start_csv_B, end_csv_B + 1)): 
        if jonas:
            all_files_B[index] = path_fol_B + f'\SoloB_2019-06-24--08-14-24_{j}.csv'
        else:
            all_files_B[index] = path_fol_B + os.path.expanduser(f'/SoloB_2019-06-24--08-14-24_{j}.csv') #need to change path_f

    vect_dict = {}
    for i in probe_list:
        #looping through each sensor
        if i < 8:
            soloA_bool = True
            all_files = all_files_A
        else:
            soloA_bool = False
            all_files = all_files_B
        if i < 9:
            num_str = f'0{i+1}'
        else: 
            num_str = i+1
        
        collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']
            
        if soloA_bool:
            df = read_files(all_files, soloA_bool, jonas, sampling_freq, collist, day=2, start_dt = start_dt, end_dt = end_dt)
            rotate_mat = rotate_24(soloA_bool)[i]
        else:
            df = read_files(all_files, soloA_bool, jonas, sampling_freq, collist, day=2, start_dt = start_dt, end_dt = end_dt)
            rotate_mat = rotate_24(soloA_bool)[i-8]
        df.iloc[:,0:3] = np.matmul(rotate_mat, df.iloc[:,0:3].values.T).T
        #print(len(df))
    
        df = shifttime(df, soloA_bool) # must shift MFSA data to MAG/spacecraft time
        
        df = df.between_time(start_dt.time(), end_dt.time())

        lowpass = False
        
        if lowpass:
            def butter_lowpass(cutoff, fs, order=5):
                nyq = 0.5 * fs
                normal_cutoff = cutoff / nyq
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                return b, a
            
            def butter_lowpass_filter(data, cutoff, fs, order=5):
                b, a = butter_lowpass(cutoff, fs, order=order)
                y = lfilter(b, a, data)
                return y

            cutoff = 15
            fs = sampling_freq

            for axis in ['X','Y','Z']:
                df[f'Probe{num_str}_{axis}'] = butter_lowpass_filter(df[f'Probe{num_str}_{axis}'], cutoff, fs)


        step_dict = calculate_dB(df, collist, peak_datetimes, start_dt, end_dt)

        X = spstats.linregress(current_dif, step_dict.get(f'Probe{num_str}_X'))
        Y = spstats.linregress(current_dif, step_dict.get(f'Probe{num_str}_Y'))
        Z = spstats.linregress(current_dif, step_dict.get(f'Probe{num_str}_Z'))
        
        if plot:
            plt.figure()
            plt.errorbar(current_dif, step_dict.get(f'Probe{num_str}_X'), yerr = step_dict.get(f'Probe{num_str}_X err'), fmt = 'bs',label = f'X grad: {round(X.slope,3)} ± {round(X.stderr,3)}', markeredgewidth = 2)
            plt.errorbar(current_dif, step_dict.get(f'Probe{num_str}_Y'), yerr = step_dict.get(f'Probe{num_str}_Y err'), fmt = 'rs', label = f'Y grad: {round(Y.slope,3)} ± {round(Y.stderr,3)}', markeredgewidth = 2)
            plt.errorbar(current_dif, step_dict.get(f'Probe{num_str}_Z'), yerr = step_dict.get(f'Probe{num_str}_Z err'), fmt = 'gs', label = f'Z grad: {round(Z.slope,3)} ± {round(Z.stderr,3)}', markeredgewidth = 2)

            plt.plot(current_dif, X.intercept + X.slope*current_dif, 'b-')
            plt.plot(current_dif, Y.intercept + Y.slope*current_dif, 'r-')
            plt.plot(current_dif, Z.intercept + Z.slope*current_dif, 'g-')

            plt.legend(loc="best")
            plt.title(f'{instrument} - Probe {num_str} - MFSA')
            plt.xlabel('dI [A]')
            plt.ylabel('dB [nT]')
            plt.show()
  
        vect_dict[f'{i+1}'] = [X.slope, Y.slope, Z.slope]

    return vect_dict

if __name__ == "__main__":
    jonas = True

    dict_current = current_peaks(jonas, plot=False)
    instrument = 'PHI'
    peak_datetimes = dict_current.get(f'{instrument} Current [A]')
    print(peak_datetimes[0], peak_datetimes[-1])
    current_dif = dict_current.get(f'{instrument} Current [A] dI')
    probes = [2,7,8]
    dB(peak_datetimes, instrument, current_dif, jonas, probes, plot=True)
