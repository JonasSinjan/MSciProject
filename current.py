#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:35:24 2019

@author: katiesimkins
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import soloA, soloB, read_files, powerspecplot, rotate_21, which_csvs
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time
from datetime import datetime
import glob

def current(jonas, plot = False, sample = False):
    if jonas:
        filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 2 Payload LCL Current Profiles.xlsx'
    else:
        filename = os.path.expanduser("~/Documents/MSciProject/Data/LCL_Data/Day_2_Payload_LCL_Current_Profiles.xlsx")

    df =  pd.read_excel(filename)
    df.set_index(['EGSE Time'], inplace = True)
    df = df.resample(f'{10}s').mean()
    #print (df.tail())

    sample = False
    if sample:
        #df = df.resample(f'{10}s').mean()
        df2 = df.loc[:,'EUI Current [A]':].groupby(np.arange(len(df))//10).mean()
        print (df2.head())

    def find_peak_times(dict, df, plot = False, i = 1):
        current_dif = np.array(df[col].diff())
        current_dif_nona = df[col].diff().dropna()
        current_dif_std = np.std(current_dif_nona)
        index_list, = np.where(abs(current_dif) > 3.3*current_dif_std) #mean is almost zero so ignore
        peak_times = [df.index[i].time() for i in index_list]
        #sorting peak times
        #for j in range(len(peak_times)):
            #if peak_times[j+1]-peak_times[j] < datetime.timedelta(minutes = 1): #time between timestamps < 30 seconds
            #if peak_times[j+1]-peak_times[j] < datetime.timedelta(seconds = 30):
            #    if peak_times[j+1]<peak_times[j]:
            #        peak_times.remove(peak_times[j+1])
            #    else:
            #        peak_times.remove(peak_times[j])
            #print(peak_times[j+1]-peak_times[j])
                
        #print(peak_times)
        print("size = ", index_list.size)
        print("std = ",current_dif_std)
        print(type(peak_times[0]))
        if str(col) not in dict.keys():
            dict[str(col)] = peak_times
            
        if plot:
            plt.figure(i)
            plt.plot(df.index.time, df[col], label=str(col))
            plt.legend(loc='best')
            plt.xlabel('Time [H:M:S]')
            plt.ylabel('Current [A]')
            
            plt.plot(df.index.time, current_dif, label='Gradient')

            #print(t)
            plt.scatter(peak_times, current_dif[index_list])
            plt.show()
            
        return dict, i
        
    dict = {}
    
    if plot != True:
        for col in df.columns:
            dict = find_peak_times(dict, df)


    if plot:
        i=1
        for col in df.columns:
            dict  = find_peak_times(dict, df, i)
            i += 1


    return dict


dict = current(False, plot = True)
#print(dict['MAG Current [A]'])