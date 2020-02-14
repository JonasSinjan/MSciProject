import pandas as pd 
import csv
import os
from current import current_peaks
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def plot_raw(windows, inst, daynumber):

    if windows:
            if daynumber == 1:
                filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 1 Payload LCL Current Profiles.xlsx'
            if daynumber == 2:
                filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 2 Payload LCL Current Profiles.xlsx'
    else:
        filename = os.path.expanduser(f"~/Documents/MSciProject/Data/LCL_Data/Day_{daynumber}_Payload_LCL_Current_Profiles.xlsx")
        
    df =  pd.read_excel(filename)
    df.set_index(['EGSE Time'], inplace = True)

    dict_current = current_peaks(windows, daynumber, plot=False) #need to get the peak datetimes
    peak_datetimes = dict_current.get(f'{inst} Current [A]')
    if daynumber == 1:
        peak_datetimes = [peak for peak in peak_datetimes if peak < datetime(2019,6,21,14,44)]

    df = df.between_time((peak_datetimes[0]-timedelta(seconds = 30)).time(), (peak_datetimes[-1]+timedelta(seconds = 30)).time())

    plt.figure()
    plt.plot(df.index.time, df[f'{inst} Current [A]'], label=str(f'{inst} Current [A]'))     
    plt.legend(loc='best')
    plt.xlabel('Time [H:M:S]')
    plt.ylabel('Current [A]')
    plt.title(f'{inst} Raw Current Profile - Day {daynumber}')
    plt.show()

if __name__ == "__main__":
    windows = True
    day = 2
    inst = 'METIS'

    plot_raw(windows, inst, day)