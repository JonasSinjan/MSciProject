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

def current(jonas, plot = False):
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

    dict = {}
    if plot != True:
    for col in df.columns:
            current_dif = np.array(df[col].diff())
            current_dif_nona = df[col].diff().dropna()
            current_dif_std = np.std(current_dif_nona)
            index_list, = np.where(current_dif > 3*current_dif_std) #mean is almost zero so ignore
            peak_times = [df.index[i] for i in index_list]
            print(peak_times)
            print(index_list.size)
            print("std = ",current_dif_std)
            if str(col) not in dict.keys():
                dict[str(col)] = peak_times


    if plot:
        i=1
<<<<<<< HEAD
        plt.figure()
        
        
        
        
=======
>>>>>>> 95239221e6126c5ab0efee1b713cce62e3565156
        for col in df.columns:
            
            current_dif = np.array(df[col].diff())
            current_dif_nona = df[col].diff().dropna()
            current_dif_std = np.std(current_dif_nona)
<<<<<<< HEAD
            current_dif_mean = np.mean(current_dif_nona)
            
            print(current_dif)
=======
            #current_dif_mean = np.mean(current_dif_nona)
            index_list, = np.where(current_dif > 3*current_dif_std) #mean is almost zero
            peak_times = [df.index[i] for i in index_list]
            print(peak_times)
            print(index_list.size)
            print("std = ",current_dif_std)
            if str(col) not in dict.keys():
                dict[str(col)] = peak_times
>>>>>>> 95239221e6126c5ab0efee1b713cce62e3565156

            plt.figure(i)
            plt.plot(df.index.time, df[col], label=str(col))
            plt.legend(loc='best')
            plt.xlabel('Time [H:M:S]')
            plt.ylabel('Current [A]')
            
            print(type(peak_times[0]))
            plt.plot(df.index.time, current_dif, label='Gradient')
            #t = [elem.to_datetime() for elem in peak_times]
            #print(t)
            #plt.scatter(t, current_dif[index_list])
            i+=1
            plt.show()

    return dict

current(True)
