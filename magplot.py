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

def mag(filepath, number_rows):

    df = pd.read_csv(filepath, header = None, nrows = number_rows)

    df.columns = ['time','X','Y','Z']

    plot = True
    if plot:
        plt.figure()
        cols = df.columns.tolist()
        for col in cols[1:]:
            plt.plot(df.index, df[col], label =f'{col}')
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
        print(time_list[2]- time_list[0], time_list[1]-time_list[2])

        plt.show()
    
if __name__ == "__main__":
    day = 1
    windows = False
    

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
        
    start_dt = datetime(2019,6,21,9,0)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
    end_dt = datetime(2019,6,21,11,0)# this is the end

    number_rows = 10000000

    mag(filepath, number_rows)
