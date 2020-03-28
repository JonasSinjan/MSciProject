import pandas as pd 
import csv
import os
from current import current_peaks
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

def plot_raw(windows, inst, daynumber, plot = False):

    if windows:
            if daynumber == 1:
                filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 1 Payload LCL Current Profiles.xlsx'
            if daynumber == 2:
                filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 2 Payload LCL Current Profiles.xlsx'
    else:
        filename = os.path.expanduser(f"~/Documents/MSciProject/Data/LCL_Data/Day_{daynumber}_Payload_LCL_Current_Profiles.xlsx")
        
    df =  pd.read_excel(filename)
    df.set_index(['EGSE Time'], inplace = True)

    if plot:
        dict_current = current_peaks(windows, daynumber, plot=False) #need to get the peak datetimes
        peak_datetimes = dict_current.get(f'{inst} Current [A]')
        if daynumber == 1:
            peak_datetimes = [peak for peak in peak_datetimes if peak < datetime(2019,6,21,14,44)]

        df = df.between_time((peak_datetimes[0]-timedelta(seconds = 30)).time(), (peak_datetimes[-1]+timedelta(seconds = 30)).time())

        plt.figure()
        plt.plot(df.index.time, df[f'{inst} Current [A]'], color = u'#1f77b4', label=str(f'{inst} Current [A]'))
        plt.scatter(df.index.time, df[f'{inst} Current [A]'], color = u'#1f77b4', s = 10, label = '_nolegend_')

        x_1 = pd.date_range(datetime(2019,6,24,9,28,36), datetime(2019,6,24,9,29,6)).tolist()
        x_2 = pd.date_range(datetime(2019,6,24,9,29,16), datetime(2019,6,24,9,29,46)).tolist()
        y = np.linspace(0,0.9,1000)
        plt.plot([datetime(2019,6,24,9,29,6).time() for i in y], y, linestyle="--", color = 'black')
        plt.plot([datetime(2019,6,24,9,28,36).time() for i in y], y, linestyle="--", color = 'black')
        plt.plot([datetime(2019,6,24,9,29,16).time() for i in y], y, linestyle="--", color = 'black')
        plt.plot([datetime(2019,6,24,9,29,46).time() for i in y], y, linestyle="--", color = 'black')        

        plt.axvspan(datetime(2019,6,24,9,28,36).time(),datetime(2019,6,24,9,29,6).time(), color = "grey", alpha = 0.3)     
        plt.axvspan(datetime(2019,6,24,9,29,16).time(),datetime(2019,6,24,9,29,46).time(), color = "grey", alpha = 0.3)  
        diff = df[f'{inst} Current [A]'].diff()
        plt.plot(df.index.time, diff)
        plt.legend(loc='best')
        plt.xlabel('Time [H:M:S]')
        plt.ylabel('Current [A]')
        plt.title(f'{inst} Raw Current Profile - Day {daynumber}')
        plt.show()
    else:
        return df
    #print (df.head())
    #print (df.tail())

if __name__ == "__main__":
    windows = True
    day = 2
    inst = 'EUI'

    plot_raw(windows, inst, day, plot = True)