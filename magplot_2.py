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
import csv
from plot_raw_current import plot_raw
import scipy.optimize as spo
import scipy.stats as spstats
import seaborn as sns

def mag(windows, day, start_dt=None, end_dt=None, plot = False, current_v_b = False):

    if day == 1:
        if windows:
            filepath = r'C:\Users\jonas\MSci-Data\PoweredDay1.csv'
        else:
            filepath = os.path.expanduser("~/Documents/MSciProject/Data/mag/PoweredDay1.csv")
        
    if day == 2:
        if windows:
            filepath = r'C:\Users\jonas\MSci-Data\PoweredDay2.csv.txt'
        else:
            filepath = os.path.expanduser("~/Documents/MSciProject/Data/mag/PoweredDay2.csv.txt")

    origin = datetime(2019, 6, 24, hour = 7, minute = 48, second = 19)

    if start_dt == None:
        start_dt = origin
        skiprows = 0
        dtime = start_dt - origin
    else:
        assert start_dt >= origin
        dtime = start_dt - origin
        skiprows = int(dtime.total_seconds()*128 - 0.518/(1/128))
    
    if end_dt != None:
        assert (end_dt-origin).total_seconds()*128 <= 3332096 #making sure the end_dt time is within the file
        des_time = end_dt - start_dt
        nrows = int(des_time.total_seconds()*128 - 0.518/(1/128))
    else:
        nrows = int(3332096 - int(dtime.total_seconds()*128 - 0.518/(1/128)))


    df = pd.read_csv(filepath, header = None, skiprows = skiprows, nrows = nrows)

    df.columns = ['time','X','Y','Z']
    df['time'] = df['time'] + 0.518


    #df.index = df.index*(1/128) #hardcoding 128 vectors/second
    df.index = pd.to_datetime(df['time'], unit = 's', origin = origin) #microsecond not added because when to_datetime unit must be 's' (due to data format of csv)
    df = df.loc[:, 'X':]
    print(df.head())
    print(df.tail())
    
    df = df.resample('1s').mean()
    
    df['X'] = df['X'] - df['X'].mean()
    df['Y'] = df['Y'] - df['Y'].mean()
    df['Z'] = df['Z'] - df['Z'].mean()
    df2 = df
    
    if plot:
        plt.figure()
        cols = df.columns.tolist()
        #df = df.resample('2s').mean()
        #tmp=[]
        if current_v_b:

            current_df = plot_raw(True, 'EUI', day, plot=False)
            current_df = current_df.between_time(start_dt.time(), end_dt.time())
            current_df = current_df.resample('1s').mean()

            print(len(current_df), len(df2))
            df2 = df2.iloc[:min(len(current_df), len(df2))]
            current_df = current_df.iloc[:min(len(current_df), len(df2))]
            xdata = current_df[f'EUI Current [A]']

            for col in cols:
                plt.scatter(current_df[f'EUI Current [A]'], df2[col], label = str(col))
            plt.xlabel('Current [A]')

            def line(x,a,b):
                return a*x + b

            params_x,cov_x = spo.curve_fit(line, xdata, df2['X'])
            params_y,cov_y = spo.curve_fit(line, xdata, df2['Y'])
            params_z,cov_z = spo.curve_fit(line, xdata, df2['Z'])

            perr_x = np.sqrt(np.diag(cov_x))
            perr_y = np.sqrt(np.diag(cov_y))
            perr_z = np.sqrt(np.diag(cov_z))
            
            #print('spo.curve_fit')
            #print('Slope = ', round(params_x[0],2), '+/-', round(perr_x[0],2))#, 'Intercept = ', params_x[1], '+/-', perr_x[1])
            #print('Slope = ', round(params_y[0],2), '+/-', round(perr_y[0],2))#, 'Intercept = ', params_y[1], '+/-', perr_y[1])
            #print('Slope = ', round(params_z[0],2), '+/-', round(perr_z[0],2))#, 'Intercept = ', params_z[1], '+/-', perr_z[1])
            
            plt.plot(xdata, params_x[0]*xdata + params_x[1], 'b-',label=f'{round(params_x[0],2)} +/-{round(perr_x[0],2)}')
            plt.plot(xdata, params_y[0]*xdata + params_y[1], 'r-',label=f'{round(params_y[0],2)} +/-{round(perr_y[0],2)}')
            plt.plot(xdata, params_z[0]*xdata + params_z[1], 'g-',label=f'{round(params_z[0],2)} +/-{round(perr_z[0],2)}')

        else:
            colour = [u'#1f77b4', u'#ff7f0e', u'#2ca02c']
            i = 0
            for col in cols:
                #df2[col] = df2[col] - np.mean(df2[col])
                #plt.plot(df2.index.time, df2[col], label =f'{col}')
                sns.lineplot(df.index.time, y = f'{col}', data = df2, label = f'{col}', color = colour[i])
                i += 1
                #var_1hz = np.std(df2[col])
                #print('std - 1Hz', col, var_1hz)
                #tmp.append(var_1hz)
           
            plt.xlabel('Time [H:M:S]')

        plt.ylabel('dB [nT]')
        plt.legend(loc="best")
        plt.title('EUI - MAG - Day 2')
        
        """
        #finding the calibration spikes - only appropriate if not in the current time span when the spikes occur (at beginning of data - but first 3 spikes seen are MAG changing measurement ranges)
        time_list = []
        df2 = df.abs()
        for col in cols:
            time_list.append(df2[col].idxmax())
        print(time_list)
        print(time_list[2]- time_list[0], time_list[1]-time_list[2])
        """
        plt.show()
    else:
        return df2
        
    #return tmp
    
if __name__ == "__main__":
    day = 2
    windows = True
    
        
    #start_dt = datetime(2019,6,24,7,55)# this is the start of the time we want to look at, #datetime(2019,6,21,10,57,50)
    #end_dt = datetime(2019,6,24,8,0)# this is the end

    start_dt = datetime(2019,6,24,9,24)
    end_dt = datetime(2019,6,24,10,9)

    b_noise = mag(windows, day, start_dt=start_dt, end_dt=end_dt, plot = True, current_v_b = False)

    """
    w = csv.writer(open(f"day2_mag_vars.csv", "w"))
    w.writerow(["Bx_var","By_var","Bz_var"])
    val = b_noise
    w.writerow([val[0],val[1],val[2]])#,val[9],val[10],val[11]])
    """
        
