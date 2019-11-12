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

def current():
    filename = os.path.expanduser("~/Documents/MSciProject/Data/LCL_Data/Day_2_Payload_LCL_Current_Profiles.xlsx")

    df =  pd.read_excel(filename)
    print (df.head())
    
    
    plot = True
    if plot:
        plt.figure()
        for col in df.columns[1:]:
            plt.plot(df['EGSE Time'], df[col], label=str(col))
            plt.legend()

            
        
        
current()
