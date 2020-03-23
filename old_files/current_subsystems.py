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

def current_peaks(windows, daynumber, plot = False, sample = False):
    #daynumber = 1
    if windows:
        if daynumber == 1:
            filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 1 Platform LCL Current Profiles.xlsx'
        if daynumber == 2:
            filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 2 Platform LCL Current Profiles.xlsx'
    else:
        filename = os.path.expanduser(f"~/Documents/MSciProject/Data/LCL_Data/Day_{daynumber}_Platform_LCL_Current_Profiles.xlsx")
    
    df =  pd.read_excel(filename)
    df.set_index(['EGSE Time'], inplace = True)
    df = df.resample(f'{5}s').mean()

    sample = False
    if sample:
        #df = df.resample(f'{10}s').mean()
        df2 = df.loc[:,'SSMM-IO [A]':].groupby(np.arange(len(df))//10).mean()
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
        
        index_list, = np.where(abs(current_dif) > 3.5*current_dif_std) #mean is almost zero so ignore

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
            if col == "SSMM-IO [A]":
                noise = [7,6,4]
            elif col == "SSMM-MC [A]":
                noise = [1,2,3]
            elif col == "STR [A]":
                noise = [1,2,3]
            elif col == "WDE-1 [A]":
                noise = [0,2,3,4,5,6,7,8,9,10,11,12]
            elif col == "WDE-2 [A]":
                noise = [0,2,3,4,5,6,7,8,9,10,11,12]
            elif col == "WDE-3 [A]":
                noise = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
            elif col == "WDE-4 [A]":
                noise = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
                peak_datetimes.append(datetime.strptime("2019-06-21 13:40:50", '%Y-%m-%d %H:%M:%S'))
            elif col == "IMU-1 Ch-1 [A]":
                noise = [2,3,4,5,6,7]
            elif col == "IMU-1 Ch-2 [A]":
                noise = [2,3,4,5]
            elif col == "IMU-1 Ch-3 [A]":
                noise = []
            elif col == "IMU-1 Ch-4 [A]":
                noise = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            elif col == "SADE [A]":
                noise = [1,2,3,4,5,6]
            elif col == "DST-1 [A]":
                noise = []
            elif col == "EPC-1_1 [A]":
                noise = []
            elif col == "EPC-1_2 [A]":
                noise = []
            elif col == "HTR3_GR4 [A]":
                noise = []
            elif col == "HTR3_GR5 [A]":
                noise = []
            elif col == "HTR3_GR6 [A]":
                noise = []
        
        
        if daynumber == 2:
            if col == "SSMM-IO [A]":
                noise = [5,7]
            elif col == "SSMM-MC [A]":
                noise = [1,2]
            elif col == "WDE-3 [A]":
                noise = [0,1,2,3,5,6,7,8,9,11]
        

        for index in sorted(noise, reverse=True):
                del peak_datetimes[index]
                
        
        index_list = np.delete(index_list, noise)

        peak_datetimes.sort()    

        print("size = ", index_list.size)
        print("std = ",current_dif_std)
        #print(type(peak_datetimes[0]))

        ### Calculating the average either side for 30 seconds ###
        
        time_to_avg = 30 + 2
        

        """
        
        ####### FOR MFSA ########
        for l, time in enumerate(peak_datetimes):
            if daynumber == 2:
                if peak_datetimes[l] > datetime.strptime("2019-06-24 17:01:00", '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283):
                    peak_datetimes.remove(peak_datetimes[l])
                    index_list = np.delete(index_list, [l])
                elif peak_datetimes[l] < datetime.strptime("2019-06-24 08:14:24", '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283):
                    peak_datetimes.remove(peak_datetimes[l])
                    index_list = np.delete(index_list, [l])
            elif daynumber == 1:
                if peak_datetimes[l] > datetime.strptime("2019-06-21 16:35:00", '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 30, milliseconds = 137):
                    peak_datetimes.remove(peak_datetimes[l])
                    index_list = np.delete(index_list, [l])
                elif peak_datetimes[l] < datetime.strptime("2019-06-21 08:09:10", '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 30, milliseconds = 137):
                    peak_datetimes.remove(peak_datetimes[l])
                    index_list = np.delete(index_list, [l])
        """
        remove_list = []
        ####### FOR MAG ########
        for l, time in enumerate(peak_datetimes):
            if daynumber == 2:
                if peak_datetimes[l] > datetime.strptime("2019-06-24 17:02:00", '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 14, milliseconds = 283):
                    remove_list.append(l)
                elif peak_datetimes[l] < datetime(2019,6,24, hour = 7, minute = 48, second = 19):
                    remove_list.append(l)
            elif daynumber == 1:
                if peak_datetimes[l] > datetime.strptime("2019-06-21 16:35:00", '%Y-%m-%d %H:%M:%S') - pd.Timedelta(days = 0, hours = 1, minutes = 59, seconds = 30, milliseconds = 137):
                    remove_list.append(l)
                elif peak_datetimes[l] < datetime(2019,6,21, hour = 8, minute = 57, second = 4):
                    remove_list.append(l)
        
        for l in sorted(remove_list, reverse=True):
            peak_datetimes.remove(peak_datetimes[l])
            index_list = np.delete(index_list, [l])
        
        


        step_list = [0]*len(peak_datetimes)
        step_err_list = [0]*len(peak_datetimes)


        for l, time in enumerate(peak_datetimes): #looping through the peaks datetimes        
            if l == 0:
                time_before_left = time - pd.Timedelta(seconds = time_to_avg)
            else:
                #time_before_left = peak_datetimes[l-1] + pd.Timedelta(seconds = 2) #old method to average over maximum possible time
                tmp = time - pd.Timedelta(seconds = time_to_avg)
                if tmp > peak_datetimes[l-1] + pd.Timedelta(seconds = 2): #checking to see which is later, if time distance between two peaks less than a minute
                    time_before_left = tmp
                else:
                    time_before_left = peak_datetimes[l-1] + pd.Timedelta(seconds = 2)
                
            time_before_right = time - pd.Timedelta(seconds = 2) #buffer time since sampling at 5sec, must be integers
            time_after_left = time + pd.Timedelta(seconds = 2)
            
            if l == len(peak_datetimes)-1:
                time_after_right = time + pd.Timedelta(seconds = time_to_avg)
            else:
                #time_after_right = peak_datetimes[l+1] - pd.Timedelta(seconds = 2) # old method to average over maximum possible time
                tmp = time + pd.Timedelta(seconds = time_to_avg)
                if tmp < peak_datetimes[l+1] - pd.Timedelta(seconds = 2):
                    time_after_right = tmp
                else:
                    time_after_right = peak_datetimes[l+1] - pd.Timedelta(seconds = 2)

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
                print(l, time)
                print(time_before_left, time_before_right)
                print(time_after_left, time_after_right)
                
            step_list[l] = step_tmp
            step_err_list[l] = step_tmp_err


        
        ###########################################################

        if str(col) not in dict.keys():
            dict[str(col)] = peak_datetimes
            #dict[str(col) + ' dI'] = current_dif[index_list]
            dict[str(col) + ' dI'] = step_list
            
        if plot:
            plt.figure(i)
         
            #print(peak_datetimes)
            peak_times = [i.time() for i in peak_datetimes]
            #print("len(peak_times) = ", len(peak_times))
            #print("len(index_list) = ", len(index_list))
            #plt.scatter(peak_times, current_dif[index_list], label='Current Step Changes')
            plt.scatter(peak_times, step_list, label='Current Step Changes')

            df2 = df.between_time((peak_datetimes[0]-pd.Timedelta(minutes = 1)).time(), (peak_datetimes[-1]+pd.Timedelta(minutes = 1)).time())
            plt.plot(df2.index.time, df2[col], label=str(col))
            
            
            plt.plot(df2.index.time, df2['Current Dif'], label='Gradient')
            plt.legend(loc='best')
            plt.xlabel('Time [H:M:S]')
            plt.ylabel('Current [A]')
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
    windows = False
    daynumber = 1
    dict_cur = current_peaks(windows, daynumber, plot = True)

    for inst in ['SSMM-IO', 'SSMM-MC', 'WDE-3']:
        peak_datetimes = dict_cur.get(f'{inst} [A]')
        print(peak_datetimes[0], peak_datetimes[-1])

