import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import soloA, soloB, read_files, powerspecplot, rotate_21, which_csvs, rotate_24, shifttime
from pandas.plotting import register_matplotlib_converters
from current import current
register_matplotlib_converters()
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time
from datetime import datetime

def dB(peak_datetimes, instrument, current_dif, jonas = True): #for only one instrument

    if jonas:
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_two\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_two\B'
    else:
        path_fol_A = os.path.expanduser("~/Documents/MSciProject/Data/day_two/A")
        path_fol_B = os.path.expanduser("~/Documents/MSciProject/Data/day_two/B")
    
    start_dt = peak_datetimes[0]-pd.Timedelta(minutes = 1)
    end_dt = peak_datetimes[1]+pd.Timedelta(minutes = 1)
    
    day = 2 #second day
    sampling_freq = 1000 #want the 1kHz
    

    for i in range(12):
        #looping through each sensor
        if i < 8:
            soloA_bool = True
        else:
            soloA_bool = False
        
        if i < 9:
            num_str = f'0{i+1}'
        else: 
            num_str = i+1
            
        start_csv, end_csv = which_csvs(soloA_bool, day ,start_dt, end_dt, tz_MAG = True)
        
        print(start_csv, end_csv)
        
        all_files = [0]*(end_csv + 1 - start_csv)
        
        collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']
        
        for index, j in enumerate(range(start_csv, end_csv + 1)): #this will loop through and add the csv files that contain the start and end time set above

            if soloA_bool:
                if jonas:
                    all_files[index] = path_fol_A + f'\SoloA_2019-06-24--08-14-46_{j}.csv'
                else:
                    all_files[index] = path_fol_A + os.path.expanduser(f'/SoloA_2019-06-24--08-14-46_{j}.csv') #need to change path_fol_A  to the path where your A folder is
            else:
                if jonas:
                    all_files[index] = path_fol_B + f'\SoloB_2019-06-24--08-14-24_{j}.csv'
                else:
                    all_files[index] = path_fol_B + os.path.expanduser(f'/SoloB_2019-06-21--08-14-24_{j}.csv') #need to change path_f
            
        if soloA_bool:
            df = read_files(all_files, soloA_bool, jonas, sampling_freq, collist, day=1, start_dt = start_dt, end_dt = end_dt)
            rotate_mat = rotate_21(soloA_bool)[i-1]
        else:
            df = read_files(all_files, soloA_bool, jonas, sampling_freq, collist, day=1, start_dt = start_dt, end_dt = end_dt)
            rotate_mat = rotate_21(soloA_bool)[i-9]
        df.iloc[:,0:3] = np.matmul(rotate_mat, df.iloc[:,0:3].values.T).T
        print(len(df))
    
        df = shifttime(df, soloA_bool) # must shift MFSA data to MAG/spacecraft time
        
        df = df.between_time(start_dt.time(), end_dt.time())
        
        #now have a df that only spans when the instrument is on
        #now need to loop through all the peak datetimes and average either side and then calculate the step change
        #then save that value to a list/array
        step_dict = {}
        for k in collist[1:]: #looping through x, y, z
            
            if str(k) not in step_dict.keys():
                step_dict[str(k)] = 0
                
            tmp_step_list = [0]*len(peak_datetimes)
            
            for time, l in enumerate(peak_datetimes): #looping through the peaks datetimes
                
                time_before_left = time - pd.Timedelta(seconds = 6)
                time_before_right = time - pd.Timedelta(seconds = 3) #buffer time since sampling at 5sec, must be integers
                time_after_left = time + pd.Timedelta(seconds = 3)
                time_after_right = time + pd.Timedelta(seconds = 6)
                
                avg_tmp = df[i][time_before_left: time_before_right].mean()
                avg_after_tmp = df[i][time_after_left:time_after_right].mean()
                
                step_tmp = avg_after_tmp - avg_tmp
                tmp_step_list[l] = step_tmp
                
            step_dict[str(k)] = tmp_step_list
        
        plt.figure()
        plt.plot(current_dif, step_dict.get(f'Probe{num_str}_X'), label = 'X') #also need to save the change in current
        plt.plot(current_dif, step_dict.get(f'Probe{num_str}_Y'), label = 'Y')
        plt.plot(current_dif, step_dict.get(f'Probe{num_str}_Z'), label = 'Z')
        plt.legend(loc="best")
        plt.title(f'{instrument} - Probe {num_str}')
        plt.show()
                
        #each sensor will have 3 lines for X, Y, Z
        
dict = current(True, plot=False)
peak_datetimes_MAG = dict.get('MAG Current [A]')
current_dif_MAG = dict.get('MAG Current [A] dI')
MAG = 'MAG'
dB(peak_datetimes_MAG, MAG, current_dif_MAG, True)

#atm get start_csv which is -8, because the MFSA data has not been shifted