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
    
    origin = datetime(2019,6,24, hour = 7, minute = 48, second = 19, microsecond=518000)
    if start_dt == None:
        start_dt = origin
        skiprows = 0
        nrows = 3332096
        end_dt = None
    
    else:
        assert start_dt >= origin
        assert (end_dt-origin).total_seconds()*128 <= 3332096 #making sure the end_dt time is within the file

        dtime = start_dt - origin
        skiprows = int(dtime.total_seconds()*128)
        nrows = int((end_dt-start_dt).total_seconds()*128)


    df = pd.read_csv(filepath, header = None, skiprows = skiprows, nrows = nrows)
    df.columns = ['time','X','Y','Z']
    df['time'] = df['time'] + 0.518

    #df.index = df.index*(1/128) #hardcoding 128 vectors/second
    df.index = pd.to_datetime(df['time'], unit = 's', origin = start_dt) #microsecond not added because when to_datetime unit must be 's' (due to data format of csv)
    df = df.loc[:, 'X':]
    print(df.head())

    plot = True
    if plot:
        plt.figure()
        #df2 = df#.iloc[2000000:]
        cols = df.columns.tolist()
        for col in cols:
            plt.plot(df.index.to_pydatetime(), df[col], label = f'{col}')
        plt.xlabel('Time')
        plt.ylabel('B [nT]')
        plt.legend(loc="best")
        plt.show()

    if start_dt == None and end_dt == None:
        first_peak = df.iloc[38500:38750].abs()
        second_peak = df.iloc[39800:40000].abs()
        third_peak = df.iloc[41200:41500].abs()
        peak_list = [first_peak, second_peak, third_peak]
        time_list = []
        for i in peak_list:
            #print(i['X'].idxmax())
            time_list.append(i['X'].idxmax())
        print(time_list)
        print(time_list[1]- time_list[0], time_list[2]-time_list[1])
    
if __name__ == "__main__":
    
    jonas = True
    
    if jonas:
        filepath = r'C:\Users\jonas\MSci-Data\PoweredDay2.csv.txt'
    else:
        filepath = os.path.expanduser("~/Documents/MSciProject/Data/mag/Day2MAGBurst.csv")
        
    start_dt = datetime(2019,6,24,9,37)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
    end_dt = datetime(2019,6,24,9,39)# this is the end

    mag(filepath, start_dt=start_dt, end_dt=start_dt)