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

def current(jonas):
    if jonas:
        filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 2 Payload LCL Current Profiles.xlsx'
    else:
        filename = os.path.expanduser("~/Documents/MSciProject/Data/LCL_Data/Day_2_Payload_LCL_Current_Profiles.xlsx")

    
    df =  pd.read_excel(filename)
    df.set_index(['EGSE Time'], inplace = True)
    #df = df.resample(f'{2}s').mean()
    print (df.head())
    
    plot = True
    if plot:
        plt.figure()
        for col in df.columns[1:]:
            plt.plot(df.index.time, df[col], label=str(col))
            plt.legend(loc='best')
            plt.xlabel('Time [H:M:S]')
            plt.ylabel('Current [A]')
        plt.show()

            
current(False)
