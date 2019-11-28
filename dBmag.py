import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import soloA, soloB, read_files, powerspecplot, rotate_21, which_csvs, rotate_24, shifttime
from pandas.plotting import register_matplotlib_converters
from current import current_peaks
register_matplotlib_converters()
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time
from datetime import datetime

def dB(peak_datetimes, instrument, current_dif, jonas): #for only one instrument

    if jonas:
        mag_filepath = r'C:\Users\jonas\MSci-Data\PoweredDay2.csv.txt'
    else:
        mag_filepath = os.path.expanduser("~/Documents/MSciProject/Data/mag/PoweredDay2.csv.txt")
    
    origin = datetime(2019,6,24, hour = 7, minute = 48, second = 19)
    
    start_dt = peak_datetimes[0]-pd.Timedelta(minutes = 1)
    end_dt = peak_datetimes[-1]+pd.Timedelta(minutes = 1)

    assert start_dt >= origin
    dtime = start_dt - origin
    skiprows = int(dtime.total_seconds()*128 - 0.518/(1/128))
    assert (end_dt-origin).total_seconds()*128 <= 3332096 #making sure the end_dt time is within the file
    des_time = end_dt - start_dt
    nrows = int(des_time.total_seconds()*128 - 0.518/(1/128))
    
    day = 2 #second day
    sampling_freq = 1000 #do we want to remove the high freq noise?

    df = pd.read_csv(mag_filepath, header = None, skiprows = skiprows, nrows = nrows)
    df.columns = ['time','X','Y','Z']
    df['time'] = df['time'] + 0.518

    df.index = pd.to_datetime(df['time'], unit = 's', origin = origin) #microsecond not added because when to_datetime unit must be 's' (due to data format of csv)
    df = df.loc[:, 'X':]

    df = df.between_time(start_dt.time(), end_dt.time())

    collist = ['time','X','Y','Z']
        
    #now have a df that only spans when the instrument is on
    #now need to loop through all the peak datetimes and average either side and then calculate the step change
    #then save that value to a list/array/dict
    step_dict = {}
    for k in collist[1:]: #looping through x, y, z
        
        if str(k) not in step_dict.keys():
            step_dict[str(k)] = 0
                
        tmp_step_list = [0]*len(peak_datetimes)
        tmp_step_err_list = [0]*len(peak_datetimes)
            
        for l, time in enumerate(peak_datetimes): #looping through the peaks datetimes
                
            time_before_left = time - pd.Timedelta(seconds = 5)
            time_before_right = time - pd.Timedelta(seconds = 2) #buffer time since sampling at 5sec, must be integers
            time_after_left = time + pd.Timedelta(seconds = 2)
            time_after_right = time + pd.Timedelta(seconds = 5)
            
            avg_tmp = df[k][time_before_left: time_before_right].mean()
            avg_tmp_std = df[k][time_before_left: time_before_right].std()
            avg_after_tmp = df[k][time_after_left:time_after_right].mean()
            avg_after_tmp_std = df[k][time_after_left:time_after_right].std()
            
            step_tmp = avg_after_tmp - avg_tmp
            step_tmp_err = np.sqrt(avg_tmp_std**2 + avg_after_tmp_std**2)
            
            tmp_step_list[l] = step_tmp
            tmp_step_err_list[l] = step_tmp_err
            
            print("dB = ", step_tmp, "dI = ", current_dif[l], "time = ", time)
        step_dict[str(k)] = tmp_step_list
        step_dict[str(k) + ' err'] = tmp_step_err_list
    
    plt.figure()
    plt.scatter(current_dif, step_dict.get('X'), label = 'X') #also need to save the change in current
    plt.scatter(current_dif, step_dict.get('Y'), label = 'Y')
    plt.scatter(current_dif, step_dict.get('Z'), label = 'Z')
    plt.legend(loc="best")
    plt.title(f'{instrument}')
    plt.xlabel('dI [A]')
    plt.ylabel('dB [nT]')
    plt.show()
            
    #each sensor will have 3 lines for X, Y, Z
    

jonas = False

dict_current = current_peaks(jonas, plot=False)
instrument = 'EPD'
peak_datetimes = dict_current.get(f'{instrument} Current [A]')
print(peak_datetimes[0], peak_datetimes[-1])
current_dif = dict_current.get(f'{instrument} Current [A] dI')
dB(peak_datetimes, instrument, current_dif, jonas = False)

#atm get start_csv which is -8, because the MFSA data has not been shifted