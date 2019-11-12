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

def mag(filepath):
    
    df = pd.read_csv(filepath, header = None)
    df.columns = ['X', 'Y', 'Z']
    print(df.head())
    print(len(df))
    
    plot = False
    if plot:
        plt.figure()
        df2 = df.iloc[:1000000]
        cols = df.columns.tolist()
        for col in cols:
            plt.plot(df2.index, df2[col], label =f'{col}')
        plt.xlabel('Index')
        plt.ylabel('B [nT]')
        plt.show()
    
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
        filepath = r'C:\Users\jonas\MSci-Data\Day2MAGBurst.csv'
        
    #else:
        
    mag(filepath)