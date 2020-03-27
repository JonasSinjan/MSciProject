import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import processing
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time
from datetime import datetime
import glob
import math
import seaborn as sns

def current_peaks(windows, daynumber, plot = False, sample = False):
    #daynumber = 1
    if windows:
        if daynumber == 1:
            filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 1 Payload LCL Current Profiles.xlsx'
        if daynumber == 2:
            filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 2 Payload LCL Current Profiles.xlsx'
    else:
        filename = os.path.expanduser(f"~/Documents/MSciProject/Data/LCL_Data/Day_{daynumber}_Payload_LCL_Current_Profiles.xlsx")
    
    df =  pd.read_excel(filename)
    df.set_index(['EGSE Time'], inplace = True)
    df1 = df
    df = df.resample(f'{5}s').mean()
    #print (df.tail())

    sample = False
    if sample:
        #df = df.resample(f'{10}s').mean()
        df2 = df.loc[:,'EUI Current [A]':].groupby(np.arange(len(df))//10).mean()
        print (df2.head())

    def find_peak_times(dict, df, plot = False, i = 1):
        if daynumber == 1:
            day = datetime(2019,6,21,0,0,0)
        if daynumber == 2:
            day = datetime(2019,6,24,0,0,0)

        #.diff() finds difference between each row as it goes - change in I per sample     
        diff = df[col].diff()
        df['Current Dif'] = diff
        current_dif = np.array(diff)
        current_dif_nona = diff.dropna()
        current_dif_std = np.std(current_dif_nona)
        index_list, = np.where(abs(current_dif) > 1.5*current_dif_std) #mean is almost zero so ignore

        peak_datetimes = [datetime.combine(datetime.date(day), df.index[i].time()) for i in index_list]
        print(col)
        #print("len = ", len(peak_datetimes))

        #removing unwanted peaks
        remove_list = []
        #removing peaks that are too close to each other
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
        
        
        if daynumber == 1:
            if col == "EUI Current [A]":
                #noise = [1,7] for 5 resample and 4 std
                noise = [5,9,10] 
            elif col == "SoloHI Current [A]":
                #noise = [3,4]
                noise = list(range(2,79))
            elif col == "PHI Current [A]":
                #noise = [4,5,8,10]
                noise = [1]+list(range(3,27))
            elif col == "STIX Current [A]":
                #noise = [1]
                noise = [0,1,2,3,4,6,7,8,9,10,11,12]
            elif col == "SPICE Current [A]":
                #noise = [0,1,2,3,4,8,9,10,11] #removing the first time it was turned on into a bad operating mode
                noise = [1,4]
            elif col == "METIS Current [A]":
                #noise = list(range(3,23))
                noise = list(range(3,172))
            elif col == "MAG Current [A]":
                noise = [1,4]
            elif col == "SWA Current [A]":
                noise = [0,1,3,12,13,14,15,16,79] + list(range(18,77))
            elif col == "RPW Current [A]":
                noise = [1,2,6] + list(range(8,87))
            

        if daynumber == 2:
            if col == "SoloHI Current [A]":
                #noise = [1,7] for 5 resample and 4 std
                noise = [1,2,3,4,5,6,7,8,9,12,16,17,18,19,20,21]
            elif col == "EUI Current [A]":
                #noise = [3,4]
                noise = [3,4,10,11,12,13,14,15,16]
            elif col == "PHI Current [A]":
                #noise = [4,5,8,10]
                noise = [4,7,8,10,13,14,15,17,18,19,20,21,22,23,24]
            elif col == "STIX Current [A]":
                #noise = [1]
                noise = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,50,51,52,53,54,55,56,57,58,59]
            elif col == "SPICE Current [A]":
                #noise = [0,1,2,3,4,8,9,10,11] #removing the first time it was turned on into a bad operating mode
                noise = [0,1,2,3,4,5,8,11,12,13,14]
            elif col == "METIS Current [A]":
                #noise = list(range(3,23))
                noise = list(range(3,65))
            elif col == "MAG Current [A]":
                noise = [3,5,6,7,8]
            elif col == "SWA Current [A]":
                noise = [0,1,2,3,4,5,6,7,10,11,13,14,15,16,17,18,19,20,21,22,24,26,27,28,29,30,31,32,33,34,35,36,37,40]
            
        for index in sorted(noise, reverse=True):
                del peak_datetimes[index]
        index_list = np.delete(index_list, noise)
        


        #print(peak_times)
        #print("size = ", index_list.size)
        #print("std = ",current_dif_std)

        step_list = [0]*len(peak_datetimes)
        step_err_list = [0]*len(peak_datetimes)

        time_to_avg = 30
        buffer = 10
        time_to_avg += buffer
        for l, time in enumerate(peak_datetimes): #looping through the peaks datetimes        
            if l == 0:
                time_before_left = time - pd.Timedelta(seconds = time_to_avg)
            else:
                #time_before_left = peak_datetimes[l-1] + pd.Timedelta(seconds = 2) #old method to average over maximum possible time
                tmp = time - pd.Timedelta(seconds = time_to_avg)
                if tmp > peak_datetimes[l-1] + pd.Timedelta(seconds = buffer): #checking to see which is later, if time distance between two peaks less than a minute
                    time_before_left = tmp
                else:
                    time_before_left = peak_datetimes[l-1] + pd.Timedelta(seconds = buffer)
                
            time_before_right = time - pd.Timedelta(seconds = buffer) #buffer time since sampling at 5sec, must be integers
            time_after_left = time + pd.Timedelta(seconds = buffer)
            
            if l == len(peak_datetimes)-1:
                time_after_right = time + pd.Timedelta(seconds = time_to_avg)
            else:
                #time_after_right = peak_datetimes[l+1] - pd.Timedelta(seconds = 2) # old method to average over maximum possible time
                tmp = time + pd.Timedelta(seconds = time_to_avg)
                if tmp < peak_datetimes[l+1] - pd.Timedelta(seconds = buffer):
                    time_after_right = tmp
                else:
                    time_after_right = peak_datetimes[l+1] - pd.Timedelta(seconds = buffer)

            df_tmp = df[col]
            df_before = df_tmp.between_time(time_before_left.time(), time_before_right.time())
            avg_tmp = df_before.mean()
            std_before = df_before.std()/np.sqrt(len(df_before))
            
            df_after = df_tmp.between_time(time_after_left.time(), time_after_right.time())
            avg_after_tmp = df_after.mean()
            std_after = df_after.std()/np.sqrt(len(df_after))


            step_tmp = avg_after_tmp - avg_tmp
            step_tmp_err = np.sqrt(std_before**2 + std_after**2)

            if math.isnan(step_tmp):
                pass
                #print(l, time)
                #print(time_before_left, time_before_right)
                #print(time_after_left, time_after_right)
                
            step_list[l] = round(step_tmp,3)
            step_err_list[l] = step_tmp_err
        #print(type(peak_datetimes[0]))
        if str(col) not in dict.keys():
            dict[str(col)] = peak_datetimes
            dict[str(col) + ' dI'] = step_list#current_dif[index_list]
            
        #print(peak_datetimes)
        print(step_list)
        if plot:
            plt.figure(i)
         
            #print(peak_datetimes)
            peak_times = [i.time() for i in peak_datetimes]
            #print("len(peak_times) = ", len(peak_times))
            #print("len(index_list) = ", len(index_list))
            #plt.scatter(peak_times, current_dif[index_list], label='Current Step Changes')

            df3 = df1.between_time((peak_datetimes[0]-pd.Timedelta(minutes = 1)).time(), (peak_datetimes[-1]+pd.Timedelta(minutes = 1)).time())
            sns.lineplot(df3.index.time, df3[col], label='Current')
            
            df2 = df.between_time((peak_datetimes[0]-pd.Timedelta(minutes = 1)).time(), (peak_datetimes[-1]+pd.Timedelta(minutes = 1)).time())
            sns.lineplot(df2.index.time, df2['Current Dif'], label='Gradient')
            sns.scatterplot(peak_times, step_list, color = u'#2ca02c',  label = 'dI', s=60)
            plt.legend(loc='best')
            plt.xlabel('Time [H:M:S]')
            plt.ylabel('Current [A]')
            inst = str(col).split(' ')[0]
            plt.title(f'{inst} Current Profile')
            #else:
            #   print("no peaks detected")
            #plt.savefig('%s_dI' % str(col))
            plt.show()
            
        return dict_cur, i
        
    dict_cur = {}


    if plot != True:
        for col in df.columns:
            dict_cur, i = find_peak_times(dict_cur, df)


    if plot:
        i=1
        for col in df.columns:
            dict_cur, i  = find_peak_times(dict_cur, df, True, i)
            i += 1



    return dict_cur

if __name__ == "__main__":
    windows = True
    daynumber = 2
    dict_cur = current_peaks(windows, daynumber, plot = True)
    #print(dict_cur['EUI Current [A]'])
    #print(dict_cur['METIS Current [A]'])

    for inst in ['EUI']:#['STIX', 'METIS', 'SPICE', 'PHI', 'SoloHI', 'EUI', 'SWA', 'EPD']:
        peak_datetimes = dict_cur.get(f'{inst} Current [A]')
        print(peak_datetimes[0], peak_datetimes[-1])

