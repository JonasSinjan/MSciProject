import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import soloA, soloB, read_files, powerspecplot, rotate_21
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from align import align
import pandas as pd
import os
import numpy as np
import scipy.signal as sps
import time
from datetime import datetime
import glob


def day_one(collist, soloA_bool, num, start_dt, end_dt):
    #set this to the directory where the data is kept on your local computer
    jonas = True

    if jonas:
        file_path_A = r'C:\Users\jonas\MSci-Data\day_one\A\SoloA_2019-06-21--08-10-10_20\SoloA_2019-06-21--08-10-10_1.csv'
        file_path_B = r'C:\Users\jonas\MSci-Data\day_one\B\SoloB_2019-06-21--08-09-10_20\SoloB_2019-06-21--08-09-10_1.csv'
        path_A = r'C:\Users\jonas\MSci-Data\day_one\A\SoloA_2019-06-21--08-10-10_50'
        path_B = r'C:\Users\jonas\MSci-Data\day_one\B\SoloB_2019-06-21--08-09-10_20'
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_one\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_one\B'
    else:
        file_path_A = os.path.expanduser("~/Documents/MSciProject/Data/SoloA_2019-06-21--08-10-10_20/SoloA_2019-06-21--08-10-10_01.csv")
        file_path_B = os.path.expanduser("~/Documents/MSciProject/Data/SoloB_2019-06-21--08-09-10_20/SoloB_2019-06-21--08-09-10_01.csv")
        path_A = os.path.expanduser("~/Documents/MSciProject/Data/SoloA_2019-06-21--08-10-10_50")
        path_B = os.path.expanduser("~/Documents/MSciProject/Data/SoloB_2019-06-21--08-09-10_20")

    if soloA_bool:
        all_files = glob.glob(path_A + "\*.csv")
        all_folders = glob.glob(path_fol_A + "\*")
        print(all_folders)
        li, length = [], []
        for folder in all_folders:
            for filename in all_files:
                df =  pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';', usecols = ['time'])
                length.append(len(df))
                li.append(df)
            df = pd.concat(li, ignore_index = True, sort=True)
            print(folder, ', seconds = ', len(df)/1000, ', mins = ',len(df)/60000, ', hours = ', len(df)/3600000)
            li = []

        #df = read_files(path_A, soloA_bool, jonas, collist, day=1, start_dt = start_dt, end_dt = end_dt)
        rotate_mat = rotate_21(soloA_bool)[num-1]
    else:
        #df = read_files(path_B, soloA_bool, jonas, collist, day=1, start_dt = start_dt, end_dt = end_dt)
        rotate_mat = rotate_21(soloA_bool)[num-9]
    df.iloc[:,0:3] = np.matmul(rotate_mat, df.iloc[:,0:3].values.T).T
    print(len(df))
    
    #time_diff = align(file_path_A, file_path_B)
    #print(time_diff)
    #now need to use pd.timedelta to subtract/add this time to the datetime object column 'time' in the df
    
    plot = True

    if plot: #plotting the raw probes results
        df2 = df#.between_time('10:57:50', '10:57:51')
        plt.figure()
        for col in collist[1:]:
            plt.plot(df2.index.to_pydatetime(), df2[col], label=str(col))
            print(df2[col].idxmax())
        plt.xlabel('Time (s)')
        plt.ylabel('B (nT)')
        plt.title(f'{num} Probe {df.index.date}')
        plt.legend()
        plt.show()

    #fs = 100
    #powerspecplot(df, fs, collist)
    
    #spectogram
    """
    x = df[collist[1]][5270000:5310000]
    #fs = 200 # sampling rate
    #f, Pxx = sps.periodogram(x,fs)
    f, t, Sxx = sps.spectrogram(x,fs)#,nperseg=700)
    plt.figure()
    plt.pcolormesh(t, f, Sxx,vmin = 0.,vmax = 0.1)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectogram')
    plt.clim()
    fig = plt.gcf()
    plt.colorbar()  
    plt.show()
    
# num = '07'
# collist = ['time', f'Probe{num}_X', f'Probe{num}_Y', f'Probe{num}_Z', f'Probe{num}_||']
# soloA_bool = True
# day_one(collist, soloA_bool)
"""

if __name__ == "__main__":
    num = 4
    soloA_bool = True
    if num <10:
        num_str = f'0{num}'
    else: 
        num_str = num
    collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']
    start_dt = datetime(2019,6,21,10,57,50)
    end_dt = datetime(2019,6,21,10,57,51)
    day_one(collist, soloA_bool, num, start_dt, end_dt)




