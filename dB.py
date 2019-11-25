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

def dB(peak_datetimes, instrument, jonas = True):

    if jonas:
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_one\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_one\B'
    else:
        path_fol_A = os.path.expanduser("~/Documents/MSciProject/Data/day_one/A")
        path_fol_B = os.path.expanduser("~/Documents/MSciProject/Data/day_one/B")
    
    start_dt = peak_datetimes[0]-pd.Timedelta(minutes = 1)
    end_dt = peak_datetimes[1]+pd.Timedelta(minutes = 1)
    
    day = 2 #second day
    sampling_freq = 1000 #want the 1kHz
    

    for i in range(12):
        #looping through each sensor
        if i < 9:
            soloA_bool = True
        else:
            soloA_bool = False
        
        if i < 10:
            num_str = f'0{i}'
        else: 
            num_str = i
            
        start_csv, end_csv = which_csvs(soloA_bool, day ,start_dt, end_dt)
        
        all_files = [0]*(end_csv + 1 - start_csv)
        
        collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']
        
        for index, i in enumerate(range(start_csv, end_csv + 1)): #this will loop through and add the csv files that contain the start and end time set above

            if soloA_bool:
                if jonas:
                    all_files[index] = path_fol_A + f'\SoloA_2019-06-21--08-10-10_{i}.csv'
                else:
                    all_files[index] = path_fol_A + os.path.expanduser(f'/SoloA_2019-06-21--08-10-10_{i}.csv') #need to change path_fol_A  to the path where your A folder is
            else:
                if jonas:
                    all_files[index] = path_fol_B + f'\SoloB_2019-06-21--08-09-10_{i}.csv'
                else:
                    all_files[index] = path_fol_B + os.path.expanduser(f'/SoloB_2019-06-21--08-09-10_{i}.csv') #need to change path_f
            
        if soloA_bool:
            df = read_files(all_files, soloA_bool, jonas, sampling_freq, collist, day=1, start_dt = start_dt, end_dt = end_dt)
            rotate_mat = rotate_21(soloA_bool)[i-1]
        else:
            df = read_files(all_files, soloA_bool, jonas, sampling_freq, collist, day=1, start_dt = start_dt, end_dt = end_dt)
            rotate_mat = rotate_21(soloA_bool)[i-9]
        df.iloc[:,0:3] = np.matmul(rotate_mat, df.iloc[:,0:3].values.T).T
        print(len(df))
    
        df = df.between_time(start_dt.time(), end_dt.time())
        #now have a df that only spans when the instrument is on
        
        #now need to loop through all the peak datetimes and average either side and then calculate the step change
        #then save that value to a list/array
          
    #plot dB against dI for each sensor on the same plot, so have several lines on the plot all going through origin
    
    #use which csv to read in files given start and end time for each instrument
    
    return dict

