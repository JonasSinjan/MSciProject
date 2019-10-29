#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:55:41 2019

@author: katiesimkins
"""

import matplotlib.pyplot as plt
from read_merge import soloA, soloB, read_files
import pandas as pd
import os
import numpy as np
import scipy.signal as sps

def align():
    jonas = False


    if jonas:
        file_path_A = r'C:\Users\jonas\MSci-Data\powered\SoloA_2019-06-21--08-10-10_20\SoloA_2019-06-21--08-10-10_1.csv' #the first couple of files in some of the folders are from earlier days
        file_path_B = r'C:\Users\jonas\MSci-Data\powered\SoloB_2019-06-21--08-09-10_20\SoloB_2019-06-21--08-09-10_1.csv'
    else:
        file_path_A = os.path.expanduser("~/Documents/MsciProject/Data/SoloA_2019-06-21--08-10-10_20/SoloA_2019-06-21--08-10-10_01.csv")
        file_path_B = os.path.expanduser("~/Documents/MsciProject/Data/SoloB_2019-06-21--08-09-10_20/SoloB_2019-06-21--08-09-10_01.csv")

        
        df_A = soloA(file_path_A)
        df_B = soloB(file_path_B)
        

        df = pd.concat([df_A, df_B], axis = 1)


        collist = ['Probe01_X','Probe01_Y','Probe01_Z','Probe01_||','Probe10_X','Probe10_Y','Probe10_Z','Probe10_||']
        
        plt.figure(1)
        for col in collist:
            plt.plot(df['time'], df[col].tolist(), label=str(col))
            
        # for col in df.columns.tolist()[-4:0]:
        #     plt.plot(df['time'], df[col], label=str(col))

        plt.xlabel('Time (s)')
        plt.ylabel('B (nT)')
        plt.legend()
        plt.show()
        
       
        
        collist_A = ['Probe01_X','Probe01_Y','Probe01_Z','Probe01_||','Probe02_X','Probe02_Y']
        max_index_A = []   #empty list for max abs values for each probe
        #print (df_A.iloc[0])
        for col in collist_A:
            probe = df_A[col].tolist()
            max_index_A.append(probe.index(max(probe, key = abs)))
        peak_index_A = max(max_index_A, key=max_index_A.count)   #find the mode of the list - most commonly shared max point between probes in A
        sample_rate = df_A['time'][1]-df_A['time'][0]
        peak_time_A = peak_index_A*sample_rate
        #peak_time_A = df_A['time'][peak_index_A]    #this is the time at which the largest peak occurs in A
        print("A",peak_time_A)
        
        
        collist_B = ['Probe10_X','Probe10_Y','Probe10_Z','Probe10_||','Probe11_X','Probe11_Y']
        max_index_B = []   #empty list for max abs values for each probe
        for col in collist_B:
            probe = df_B[col].tolist()
            max_index_B.append(probe.index(max(probe, key = abs)))
        peak_index_B = max(max_index_B, key=max_index_B.count)   #find the mode of the list - most commonly shared max point between probes in A
        peak_time_B = peak_index_B*sample_rate    #this is the time at which the largest peak occurs in B
        print("B",peak_time_B)
        
        time_diff = peak_time_A - peak_time_B
        
        B_columns = ['Probe10_X','Probe10_Y','Probe10_Z','Probe10_||','Probe11_X','Probe11_Y','Probe11_Z','Probe11_||','Probe09_X','Probe09_Y','Probe09_Z','Probe09_||','Probe12_X','Probe12_Y','Probe12_Z','Probe12_||']
        
        for col in B_columns:
            df[col] = df[col].shift(int(time_diff/sample_rate))
        
        
        plt.figure(2)
        for col in collist:
            plt.plot(df['time'], df[col].tolist(), label=str(col))
            
        # for col in df.columns.tolist()[-4:0]:
        #     plt.plot(df['time'], df[col], label=str(col))

        plt.xlabel('Time (s)')
        plt.ylabel('B (nT)')
        plt.legend()
        plt.show()
        
        




align()