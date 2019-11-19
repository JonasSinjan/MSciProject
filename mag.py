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

def mag(filepath, start_dt=None, end_dt=None):
    
    origin = datetime(2019,6,24, hour = 7, minute = 48, second = 19)

    if start_dt == None:
        start_dt = origin
        skiprows = 0
        dtime = start_dt - origin
    else:
        assert start_dt >= origin
        dtime = start_dt - origin
        skiprows = int(dtime.total_seconds()*128 - 0.518/(1/128))
    
    if end_dt != None:
        assert (end_dt-origin).total_seconds()*128 <= 3332096 #making sure the end_dt time is within the file
        des_time = end_dt - start_dt
        nrows = int(des_time.total_seconds()*128 - 0.518/(1/128))
    else:
        nrows = int(3332096 - int(dtime.total_seconds()*128 - 0.518/(1/128)))


    df = pd.read_csv(filepath, header = None, skiprows = skiprows, nrows = nrows)

    df.columns = ['time','X','Y','Z']
    df['time'] = df['time'] + 0.518


    #df.index = df.index*(1/128) #hardcoding 128 vectors/second
    df.index = pd.to_datetime(df['time'], unit = 's', origin = origin) #microsecond not added because when to_datetime unit must be 's' (due to data format of csv)
    df = df.loc[:, 'X':]
    print(df.head())
    print(df.tail())

    plot = True
    if plot:
        plt.figure()
        cols = df.columns.tolist()
        for col in cols:
            plt.plot(df.index.time, df[col], label =f'{col}')
        plt.xlabel('Time [H:M:S]')
        plt.ylabel('B [nT]')
        plt.legend(loc="best")
        
    
        time_list = []
        df2 = df.abs()
        for col in cols:
            time_list.append(df2[col].idxmax())
        print(time_list)
        print(time_list[2]- time_list[0], time_list[1]-time_list[2])

        plt.show()
    
if __name__ == "__main__":
    
    jonas = True
    
    if jonas:
        filepath = r'C:\Users\jonas\MSci-Data\PoweredDay2.csv.txt'
    else:
        filepath = os.path.expanduser("~/Documents/MSciProject/Data/mag/Day2MAGBurst.csv")
        
    start_dt = datetime(2019,6,24,8,2)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
    end_dt = datetime(2019,6,24,8,3)# this is the end

    mag(filepath, start_dt=start_dt, end_dt=end_dt)
