import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import processing
from pandas.plotting import register_matplotlib_converters
from current import current_peaks
register_matplotlib_converters()
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time
from datetime import datetime
import scipy.stats as spstats
import csv 

def dB(peak_datetimes, instrument, current_dif, windows): #for only one instrument


    sampling_freq = 1

    if windows:
        mag_filepath = r'C:\Users\jonas\MSci-Data\PoweredDay2.csv.txt'
    else:
        mag_filepath = os.path.expanduser("~/Documents/MSciProject/Data/mag/PoweredDay2.csv.txt")
    
    origin = datetime(2019,6,24, hour = 7, minute = 48, second = 19)
    
    start_dt = peak_datetimes[0]-pd.Timedelta(minutes = 1)
    end_dt = peak_datetimes[-1]+pd.Timedelta(minutes = 1)

    assert start_dt >= origin
    dtime = start_dt - origin
    skiprows = int(dtime.total_seconds()*128 - 0.518/(1/128))
    assert (end_dt-origin).total_seconds()*128 <= 3332096 #making sure the end_dt time is within the file
    des_time = end_dt - start_dt
    nrows = int(des_time.total_seconds()*128 - 0.518/(1/128))
    
    #day = 2 #second day

    df = pd.read_csv(mag_filepath, header = None, skiprows = skiprows, nrows = nrows)
    df.columns = ['time','X','Y','Z']
    df['time'] = df['time'] + 0.518

    df.index = pd.to_datetime(df['time'], unit = 's', origin = origin) #microsecond not added because when to_datetime unit must be 's' (due to data format of csv)
    df = df.loc[:, 'X':]

    df = df.between_time(start_dt.time(), end_dt.time())

    if sampling_freq < 1000:
        factor = int(1000/sampling_freq)
        if factor >= 0.001:
            df = df.resample(f'{factor}ms').mean()
        else:
            print('The resampling is in the wrong units - must be factor*milliseconds')
    else:
        print('The desired sampling frequency is greater than the raw data available - defaulted to 1kHz')
        
    fs = 20

    collist = ['time','X','Y','Z']
    #df = df.resample('1s').mean()
    #processing.powerspecplot(df, fs, collist, False, inst = instrument)

    step_dict = processing.calculate_dB(df, peak_datetimes)


    X = spstats.linregress(current_dif, step_dict.get('X'))
    Y = spstats.linregress(current_dif, step_dict.get('Y'))
    Z = spstats.linregress(current_dif, step_dict.get('Z'))
    
    """
    plt.figure()
    plt.errorbar(current_dif, step_dict.get('X'), yerr = step_dict.get('X err'), fmt = 'bs', label = f'X grad: {round(X.slope,2)} ± {round(X.stderr,2)} int: {round(X.intercept, 2)}',markeredgewidth = 2) #also need to save the change in current
    plt.errorbar(current_dif, step_dict.get('Y'), yerr = step_dict.get('Y err'), fmt = 'rs', label = f'Y grad: {round(Y.slope,2)} ± {round(Y.stderr,2)} int: {round(Y.intercept, 2)}', markeredgewidth = 2)
    plt.errorbar(current_dif, step_dict.get('Z'), yerr = step_dict.get('Z err'), fmt = 'gs', label = f'Z grad: {round(Z.slope,2)} ± {round(Z.stderr,2)} int: {round(Z.intercept, 2)}', markeredgewidth = 2)

    plt.plot(current_dif, X.intercept + X.slope*current_dif, 'b-')
    plt.plot(current_dif, Y.intercept + Y.slope*current_dif, 'r-')
    plt.plot(current_dif, Z.intercept + Z.slope*current_dif, 'g-')

    plt.legend(loc="best")
    plt.title(f'{instrument} - MAG')
    plt.xlabel('dI [A]')
    plt.ylabel('dB [nT]')
    plt.show()
    
    print('sps.linregress')
    print('Slope = ', X.slope, '+/-', X.stderr, ' Intercept = ', X.intercept)
    print('Slope = ', Y.slope, '+/-', Y.stderr, ' Intercept = ', Y.intercept)
    print('Slope = ', Z.slope, '+/-', Z.stderr, ' Intercept = ', Z.intercept)
    #each sensor will have 3 lines for X, Y, Z
    """
    vect_dict = {}
    
    dBdI = {}
    for dI in range(len(current_dif)):
        dBdI[f'{dI+1}'] = [current_dif[dI],step_dict.get('X')[dI],step_dict.get('X err')[dI],step_dict.get('Y')[dI],step_dict.get('Y err')[dI],step_dict.get('Z')[dI],step_dict.get('Z err')[dI]]
                
        w = csv.writer(open(f"{instrument}_vect_dict_mag_dBdI_day2.csv", "w"))
        w.writerow(["key","dI","dB_X","dB_X_err","dB_Y","dB_Y_err","dB_Z","dB_Z_err"])
    for key, val in dBdI.items():
        w.writerow([key,val[0],val[1],val[2],val[3],val[4],val[5],val[6]])#,val[9],val[10],val[11]])

    vect_list = [instrument,X.slope, Y.slope, Z.slope,X.stderr,Y.stderr,Z.stderr, X.intercept ,Y.intercept, Z.intercept ]
    return vect_list
    

if __name__ == "__main__":
    windows = True

    dict_current = current_peaks(windows,2, plot=False)
    instru_list = ['EPD', 'EUI', 'SWA', 'STIX', 'METIS', 'SPICE', 'PHI', 'SoloHI']
    vect_dict = {}
    i = 0
    for instrument in instru_list:
        i+=1
        #get list of the peaks' datetimes for the desired instrument
        peak_datetimes = dict_current.get(f'{instrument} Current [A]')
        print(peak_datetimes[0], peak_datetimes[-1])
        #print first and last peak datetime to affirm correct instrument
        #print(peak_datetimes[0], peak_datetimes[-1]) 
        #need current dif (gradient in current) to plot later
        current_dif = dict_current.get(f'{instrument} Current [A] dI')
        #create dictionary of the Magnetic Field/Amp proportionality for the desired instrument
        vect_dict[f'{i}'] = dB(peak_datetimes, instrument, current_dif, windows)
        #write the Magnetic Field/Amp proportionality to csv
        

w = csv.writer(open("mag_vect_dict_slopes_day2.csv", "w"))
w.writerow(["instrument","X.slope_lin", "Y.slope_lin", "Z.slope_lin","X.slope_lin_err", "Y.slope_lin_err", "Z.slope_lin_err","X_zero_err","Y_zero_err","Z_zero_err"])#,"X.slope_curve", "Y.slope_curve", "Z.slope_curve","X.slope_curve_err", "Y.slope_curve_err", "Z.slope_curve_err"])
for key, val in vect_dict.items():
    w.writerow([val[0],val[1],val[2],val[3],val[4],val[5],val[6],val[7],val[8],val[9]])#,val[9],val[10],val[11]])

    
    