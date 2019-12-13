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
import scipy.optimize as spo
import matplotlib.pyplot as plt
import csv 


def dB(peak_datetimes, instrument, current_dif, jonas, probe_list, plot=False): #for only one instrument

    if jonas:
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_two\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_two\B'
    else:
        path_fol_A = os.path.expanduser("~/Documents/MSciProject/Data/day_two/A")
        path_fol_B = os.path.expanduser("~/Documents/MSciProject/Data/day_two/B")
    
    start_dt = peak_datetimes[0] - pd.Timedelta(minutes = 3)
    end_dt = peak_datetimes[-1] + pd.Timedelta(minutes = 3)
    
    day = 2 #second day
    sampling_freq = 1000 #do we want to remove the high freq noise?
    
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
        #print(df.head())
        #print(df.tail())
    
        df = shifttime(df, soloA_bool) # must shift MFSA data to MAG/spacecraft time
        
        #print(df.head())
        #print(df.tail())
        
        df = df.between_time(start_dt.time(), end_dt.time())
        
        #print(df.head())
        #print(df.tail())

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


        step_dict = calculate_dB(df, peak_datetimes)

        #adding bonus point of origin
        xdata = list(current_dif)
        xdata.append(0.0)
        
        probe_x_tmp = step_dict.get(f'Probe{num_str}_X')
        probe_y_tmp = step_dict.get(f'Probe{num_str}_Y')
        probe_z_tmp = step_dict.get(f'Probe{num_str}_Z')

        probe_x_tmp.append(0.0)
        probe_y_tmp.append(0.0)
        probe_z_tmp.append(0.0)

        probe_x_tmp_err = step_dict.get(f'Probe{num_str}_X err')
        probe_y_tmp_err = step_dict.get(f'Probe{num_str}_Y err')
        probe_z_tmp_err = step_dict.get(f'Probe{num_str}_Z err')

        probe_x_tmp_err.append(0.0) #error on bonus point should be zero, but curve_fit requires finite error - and this forces the line through the origin anyway
        probe_y_tmp_err.append(0.0)
        probe_z_tmp_err.append(0.0)

        X = spstats.linregress(xdata, probe_x_tmp) #adding bonus point has little effect on grad - only changes intercept
        Y = spstats.linregress(xdata, probe_y_tmp)
        Z = spstats.linregress(xdata, probe_z_tmp)
        
        def line(x,a):
            return a*x #forcing the line through the origin - same as adding bonus point at origin - as curve_fit requires error on origin point - which must be set to ~0 physically

        params_x,cov_x = spo.curve_fit(line, current_dif, probe_x_tmp[:-1], sigma = probe_x_tmp_err[:-1], absolute_sigma = True)
        params_y,cov_y = spo.curve_fit(line, current_dif, probe_y_tmp[:-1], sigma = probe_y_tmp_err[:-1], absolute_sigma = True)
        params_z,cov_z = spo.curve_fit(line, current_dif, probe_z_tmp[:-1], sigma = probe_z_tmp_err[:-1], absolute_sigma = True)

        perr_x = np.sqrt(np.diag(cov_x))
        perr_y = np.sqrt(np.diag(cov_y))
        perr_z = np.sqrt(np.diag(cov_z))

        print('sps.linregress')
        print('Slope = ', X.slope, '+/-', X.stderr, ' Intercept = ', X.intercept)
        print('Slope = ', Y.slope, '+/-', Y.stderr, ' Intercept = ', Y.intercept)
        print('Slope = ', Z.slope, '+/-', Z.stderr, ' Intercept = ', Z.intercept)

        print('~')
        print('spo.curve_fit')
        print('Slope & Intercept = ', params_x, '+/-', perr_x)
        print('Slope & Intercept = ', params_y, '+/-', perr_y)
        print('Slope & Intercept = ', params_z, '+/-', perr_z)

        if plot:
            plt.figure()
            plt.errorbar(xdata, probe_x_tmp, yerr = probe_x_tmp_err, fmt = 'bs',label = f'X grad: {round(X.slope,3)} ± {round(X.stderr,3)}', markeredgewidth = 2)
            plt.errorbar(xdata, probe_y_tmp, yerr = probe_y_tmp_err, fmt = 'rs', label = f'Y grad: {round(Y.slope,3)} ± {round(Y.stderr,3)}', markeredgewidth = 2)
            plt.errorbar(xdata, probe_z_tmp, yerr = probe_z_tmp_err, fmt = 'gs', label = f'Z grad: {round(Z.slope,3)} ± {round(Z.stderr,3)}', markeredgewidth = 2)

            plt.plot(xdata, X.intercept + X.slope*np.array(xdata), 'b-')
            plt.plot(xdata, Y.intercept + Y.slope*np.array(xdata), 'r-')
            plt.plot(xdata, Z.intercept + Z.slope*np.array(xdata), 'g-')

            plt.plot(current_dif, params_x[0]*current_dif, 'b:', label = f'curve_fit - X grad: {round(params_x[0],3)} ± {round(perr_x[0],3)}')
            plt.plot(current_dif, params_y[0]*current_dif, 'r:', label = f'curve_fit - Y grad: {round(params_y[0],3)} ± {round(perr_y[0],3)}')
            plt.plot(current_dif, params_z[0]*current_dif, 'g:', label = f'curve_fit - Z grad: {round(params_z[0],3)} ± {round(perr_z[0],3)}')

            plt.legend(loc="best")
            plt.title(f'{instrument} - Probe {num_str} - MFSA')
            plt.xlabel('dI [A]')
            plt.ylabel('dB [nT]')
            plt.show()
  
        vect_dict[f'{i+1}'] = [X.slope, Y.slope, Z.slope,X.stderr,Y.stderr,Z.stderr,params_x[0],params_y[0],params_z[0],perr_x[0],perr_y[0],perr_z[0]] #atm linear regression gradient - or should it be curve_fit?


    return vect_dict

if __name__ == "__main__":
    jonas = False
    dict_current = current_peaks(jonas, plot=False)
    instrument = 'STIX'
    peak_datetimes = dict_current.get(f'{instrument} Current [A]')
    print(peak_datetimes[0], peak_datetimes[-1])
    current_dif = dict_current.get(f'{instrument} Current [A] dI')
    probes = range(12)
    vect_dict = dB(peak_datetimes, instrument, current_dif, jonas, probes, plot=False)
    w = csv.writer(open(f"{instrument}_vect_dict.csv", "w"))
    w.writerow(["Probe","X.slope_lin", "Y.slope_lin", "Z.slope_lin","X.slope_lin_err", "Y.slope_lin_err", "Z.slope_lin_err","X.slope_curve", "Y.slope_curve", "Z.slope_curve","X.slope_curve_err", "Y.slope_curve_err", "Z.slope_curve_err"])
    for key, val in vect_dict.items():
        w.writerow([key,val[0],val[1],val[2],val[3],val[4],val[5],val[6],val[7],val[8],val[9],val[10],val[11]])
