import matplotlib.pyplot as plt
from processing import processing
import pandas as pd
import os
import numpy as np

def align(file_path_A, file_path_B):
    #this function should only return the time diff between soloA and soloB files
    #which should then be read in when read_merge is executed
    
    df_A = processing.soloA(file_path_A)
    df_B = processing.soloB(file_path_B)
    #df_A = df_A[df_A['time']<=600]
    print(len(df_A), len(df_B))
    #df = df_A.merge(df_B) #if want to merge
    #print(len(df)) #this cuts off the end half of soloB

    collist_A = df_A.columns.tolist()
    max_index_A = []   #empty list for max abs values for each probe
    for col in collist_A[1:]:
        #print(col)
        probe = df_A[col].abs() #creates absolute series
        max_index_A.append(probe.idxmax()) #returns first index of maximum
    print(max_index_A)
    peak_index_A = max(max_index_A, key=max_index_A.count)   #find the mode of the list - most commonly shared max point between probes in A
    peak_time_A = df_A['time'].iloc[peak_index_A] #suggest this instead-means never have to worry about sample rate
    #print("A",peak_time_A)
    
    collist_B = df_B.columns.tolist()
    max_index_B = []   #empty list for max abs values for each probe
    for col in collist_B[1:]:
        #print(col)
        probe = df_B[col].abs()
        max_index_B.append(probe.idxmax())
    peak_index_B = max(max_index_B, key=max_index_B.count) 
    print(max_index_B)  #find the mode of the list - most commonly shared max point between probes in A
    peak_time_B = df_B['time'].iloc[peak_index_B] #added time column to B files - when merging have to take this into account
    #print("B",peak_time_B)
    
    time_diff = peak_time_A - peak_time_B
    
    return time_diff
    
#df_B['time'] = df_B['time'] + time_diff
