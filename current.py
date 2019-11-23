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
    df = df.resample(f'{5}s').mean()
    #print (df.tail())

    sample = False
    if sample:
        #df = df.resample(f'{10}s').mean()
        df2 = df.loc[:,'EUI Current [A]':].groupby(np.arange(len(df))//10).mean()
        print (df2.head())

    def find_peak_times(dict, df, plot = False, i = 1):
        day = datetime(2019,6,24,0,0,0)
        current_dif = np.array(df[col].diff())
        current_dif_nona = df[col].diff().dropna()
        current_dif_std = np.std(current_dif_nona)
        index_list, = np.where(abs(current_dif) > 4*current_dif_std) #mean is almost zero so ignore

        peak_datetimes = [datetime.combine(datetime.date(day), df.index[i].time()) for i in index_list]
        print(col)
        print("len = ", len(peak_datetimes))

        #removing unwanted peaks
        remove_list = []
        for j in range(len(peak_datetimes)-1):
            if (peak_datetimes[j+1]-peak_datetimes[j]).total_seconds() < 50:
                dict_tmp = {'j': abs(current_dif[index_list[j]]), 'j+1': abs(current_dif[index_list[j+1]])}

                min_var = min(dict_tmp, key = dict_tmp.get)
                if len(remove_list) != 0:
                    if j == remove_list[-1]:
                        continue #for if j+1 removed, in next loop, j+1 becomes j and if j then removed - will be removed twice
                #print('Peak j = ', peak_datetimes[j])
                #print('Peak j+1 = ', peak_datetimes[j+1])
                #print(min_var, peak_datetimes[j])
                
                if min_var == 'j':
                    remove_list.append(j)
                else:
                    remove_list.append(j+1) 
             
        #for index, i in enumerate(remove_list):
        #    print(i,index, len(peak_datetimes))
        for index in sorted(remove_list, reverse=True):
                del peak_datetimes[index]
        index_list = np.delete(index_list, remove_list)

        noise = []

        if col == "SoloHI Current [A]":
            noise = [1,7]
        elif col == "PHI Current [A]":
            noise = [10]
        elif col == "SPICE Current [A]":
            noise = [0,1,2,3,4,8,10]
        elif col == "METIS Current [A]":
            noise = list(range(3,23))
        
        for index in sorted(noise, reverse=True):
                del peak_datetimes[index]
        index_list = np.delete(index_list, noise)
        
        #print(peak_times)
        print("size = ", index_list.size)
        print("std = ",current_dif_std)
        #print(type(peak_datetimes[0]))
        if str(col) not in dict.keys():
            dict[str(col)] = peak_datetimes
            
        if plot:
            plt.figure(i)
            plt.plot(df.index.time, df[col], label=str(col))
            plt.legend(loc='best')
            plt.xlabel('Time [H:M:S]')
            plt.ylabel('Current [A]')
            
            plt.plot(df.index.time, current_dif, label='Gradient')
            #print(peak_datetimes)
            peak_times = [i.time() for i in peak_datetimes]
            print("len(peak_times) = ", len(peak_times))
            print("len(index_list) = ", len(index_list))
            plt.scatter(peak_times, current_dif[index_list])
            #else:
            #   print("no peaks detected")
            plt.show()
            
        return dict, i
        
    dict = {}
    

    #def delta(dict,df,col):
        



    if plot != True:
        for col in df.columns:
            dict, i = find_peak_times(dict, df)


    if plot:
        i=1
        for col in df.columns:
            dict, i  = find_peak_times(dict, df, i)
            i += 1
            print(dict)




    return dict


dict = current(False, plot = True)
#print(dict['MAG Current [A]'])