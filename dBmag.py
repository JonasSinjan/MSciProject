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

    df = pd.read_csv(mag_filepath, header = None, skiprows = skiprows, nrows = nrows)
    df.columns = ['time','X','Y','Z']
    df['time'] = df['time'] + 0.518

    df.index = pd.to_datetime(df['time'], unit = 's', origin = origin) #microsecond not added because when to_datetime unit must be 's' (due to data format of csv)
    df = df.loc[:, 'X':]

    df = df.between_time(start_dt.time(), end_dt.time())
    fs = 128

    collist = ['time','X','Y','Z']

    powerspecplot(df, fs, collist, False, inst = instrument)

    step_dict = calculate_dB(df, collist, peak_datetimes, start_dt, end_dt)

    plt.figure()
    plt.errorbar(current_dif, step_dict.get('X'), yerr = step_dict.get('X err'), fmt = 'bs',label = 'X', markeredgewidth = 2) #also need to save the change in current
    X = spstats.linregress(current_dif, step_dict.get('X'))
    plt.errorbar(current_dif, step_dict.get('Y'), yerr = step_dict.get('Y err'), fmt = 'rs', label = 'Y', markeredgewidth = 2)
    Y = spstats.linregress(current_dif, step_dict.get('Y'))
    plt.errorbar(current_dif, step_dict.get('Z'), yerr = step_dict.get('Z err'), fmt = 'gs', label = 'Z', markeredgewidth = 2)
    Z = spstats.linregress(current_dif, step_dict.get('Z'))

    plt.plot(current_dif, X.intercept + X.slope*current_dif, 'b-', label = X.rvalue)
    plt.plot(current_dif, Y.intercept + Y.slope*current_dif, 'r-', label = Y.rvalue)
    plt.plot(current_dif, Z.intercept + Z.slope*current_dif, 'g-', label = Z.rvalue)

    plt.legend(loc="best")
    plt.title(f'{instrument} - MAG')
    plt.xlabel('dI [A]')
    plt.ylabel('dB [nT]')
    plt.show()
    #each sensor will have 3 lines for X, Y, Z
    
if __name__ == "__main__":
    jonas = True

    dict_current = current_peaks(jonas, plot=False)
    instrument = 'EUI'
    peak_datetimes = dict_current.get(f'{instrument} Current [A]')
    print(peak_datetimes[0], peak_datetimes[-1])
    current_dif = dict_current.get(f'{instrument} Current [A] dI')
    dB(peak_datetimes, instrument, current_dif, jonas)