import pandas as pd
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scipy.signal as sps
from datetime import datetime, timedelta
import time
import math

def mag_1(filepath):#, number_rows):

    df = pd.read_csv(filepath, header = None)#, nrows = number_rows)

    df.columns = ['time','X','Y','Z']

    origin = datetime(2019, 6, 21, hour = 10, minute = 37, second = 59)
    
    """
    start_dt = peak_datetimes[0] - pd.Timedelta(minutes = 1)
    end_dt = peak_datetimes[-1] + pd.Timedelta(minutes = 1)

    assert start_dt >= origin
    dtime = start_dt - origin
    skiprows = int(dtime.total_seconds()*128 - 0.518/(1/128))
    assert (end_dt - origin).total_seconds()*128 <= 3332096 #making sure the end_dt time is within the file
    des_time = end_dt - start_dt
    nrows = int(des_time.total_seconds()*128 - 0.518/(1/128))
    df = pd.read_csv(mag_filepath, header = None, skiprows = skiprows, nrows = nrows)
    """
    
    df['time'] = df['time'] + 0.78

    df.index = pd.to_datetime(df['time'], unit = 's', origin = origin) #microsecond not added because when to_datetime unit must be 's' (due to data format of csv)
    df = df.loc[:, 'X':]

    plot = True
    if plot:
        plt.figure()
        cols = df.columns.tolist()
        for col in cols:
            plt.plot(df.index.time, df[col], label =f'{col}')
        plt.xlabel('Time')
        plt.ylabel('B [nT]')
        plt.legend(loc="best")
        plt.title('MAG Powered Day 1')
        
        #finding the calibration spikes - only appropriate if not in the current time span when the spikes occur (at beginning of data - but first 3 spikes seen are MAG changing measurement ranges)
        time_list = []
        df2 = df.abs()
        for col in cols:
            time_list.append(df2[col].idxmax())
        print(time_list)
        print(time_list[2] - time_list[0], time_list[1] - time_list[2])

        plt.show()
    
if __name__ == "__main__":
    day = 1
    windows = True
    
    if day == 1:
        if windows:
            filepath = r'C:\Users\jonas\MSci-Data\PoweredDay1.csv'
        else:
            filepath = os.path.expanduser("~/Documents/MSciProject/Data/mag/PoweredDay1.csv")
        
    if day == 2:
        if windows:
            filepath = r'C:\Users\jonas\MSci-Data\PoweredDay2.csv.txt'
        else:
            filepath = os.path.expanduser("~/Documents/MSciProject/Data/mag/PoweredDay2.csv.txt")
        
    start_dt = datetime(2019, 6, 21, 9, 0)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
    end_dt = datetime(2019, 6, 21, 11, 0)# this is the end

    number_rows = 5000000

    mag_1(filepath)#, number_rows)
