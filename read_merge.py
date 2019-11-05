import pandas as pd
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
import os
import scipy.signal as sps
from datetime import datetime, timedelta
import time

def read_files(path, soloA, jonas, collist=None):
    #path - location of folder to concat
    #soloA - set to True if soloA, if soloB False 
    if jonas: 
        all_files = glob.glob(path + "\*.csv")
    else: 
        all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        #time = pd.read_csv(filename, skiprows = 7, nrows = 1, header = None)
        #start_time = filename.strip('-')        
        if soloA:
            if collist == None:
                df =  pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';')
                cols = df.columns.tolist()
                new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #this will reorder the columns into the correct order
                df = df[new_cols]
            else:
                df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';', usecols = collist)
        else:
            if collist == None:
                df =  pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';')
                cols = df.columns.tolist()
                new_cols = [cols[0]] + cols[9:13] + cols[1:9] + cols[13:17]
                df = df[new_cols]
            else:
                df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';', usecols = collist)
            
        li.append(df)
        
    
        
    df = pd.concat(li, ignore_index = True, sort=True)

    
    start = time.process_time()
    if soloA:
        if '21' in all_files[0]: #for day_one
            df['time'] = df['time'] + 10.12
            df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-21 08:10:00' )
        elif '24' in all_files[0]: #for day_two
            df['time'] = df['time'] + 46.93
            df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-24 08:14:00' )
    else:
        if '21' in all_files[0]:
            df['time'] = df['time'] + 10
            df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-21 08:09:00' )
        elif '24' in all_files[0]:
            df['time'] = df['time'] + 24
            df['time'] =  pd.to_datetime(df['time'], unit = 's', origin = '2019-06-24 08:14:00' )

    df['time'] = df['time'].dt.round('ms')
    df = df.sort_values('time', ascending = True, kind = 'mergesort')
    #df = df.reset_index(drop=True)
    print(time.process_time() - start)
    df.set_index('time', inplace = True)
    print(df.head())
    #print(df['time'].head())
    return df
    
def soloA(file_path):
    #skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
    df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';')
    cols = df.columns.tolist()
    new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9] #reorder the columns into the correct order
    df = df[new_cols]
    return df

def soloB(file_path):
    #skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
    df_B = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';')
    cols = df_B.columns.tolist()
    new_cols = [cols[0]] + cols[9:13] + cols[1:9] + cols[13:17]#reorder the columns into the correct order # adding time as first column
    df_B = df_B[new_cols]
    return df_B

def powerspecplot(df, fs, collist):
    
    probe_x = collist[1]
    probe_y = collist[2]
    probe_z = collist[3]
    probe_m = collist[4]
    x = df[probe_x]#[:20000]
    f_x, Pxx_x = sps.periodogram(x,fs, scaling='spectrum')
    x = df[probe_y]#[:20000]
    f_y, Pxx_y = sps.periodogram(x,fs, scaling='spectrum')
    x = df[probe_z]#[:20000]
    f_z, Pxx_z = sps.periodogram(x,fs, scaling='spectrum')
    x = df[probe_m]#[:20000]
    f_m, Pxx_m = sps.periodogram(x,fs, scaling='spectrum')
    
    def plot_power(f,Pxx,probe):
        plt.semilogy(f,np.sqrt(Pxx)) #sqrt required for power spectrum, and semi log y axis
        plt.xlim(0,60)
        plt.ylim(10e-4,10e-1)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Log(FFT magnitude)')
        plt.title(f'{probe}')
        peaks, _ = sps.find_peaks(np.log10(np.sqrt(Pxx)), prominence = 3)
        print([round(i,1) for i in f[peaks] if i <= 20], len(peaks))
        plt.semilogy(f[peaks], np.sqrt(Pxx)[peaks], marker = 'x', markersize = 10, color='orange', linestyle = 'None')
    

    plt.figure()
    plt.title('Power Spectrum')
    plt.subplot(221)
    plot_power(f_x, Pxx_x, probe_x)
    
    plt.subplot(222)
    plot_power(f_y, Pxx_y, probe_y)

    plt.subplot(223)
    plot_power(f_z, Pxx_z, probe_z)
    
    plt.subplot(224)
    plot_power(f_m, Pxx_m, probe_m)
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    

def rotate_21(soloA_bool):
    if soloA_bool:
        M_1 = np.array([-0.01151451,-0.03379413,-0.99912329,-0.99966879,0.02182397,0.01062976,0.02152466,0.99893068,-0.03362521])
        M_2 = np.array([0.0221395,-0.00814719,-1.00104386,-0.99984468,0.04330073,-0.02333449,0.04405573,1.00084827,-0.00708282])
        M_3 = np.array([0.03283952,-0.00829181,-0.99809581,-0.99786078,-0.01761458,-0.03303076,-0.01707146,0.99865858,-0.00872099])
        M_4 = np.array([0.00742014,-0.00079088,-1.00090218,-1.00039189,0.02286711,-0.00760828,0.02237911,1.00026345,-0.00006682])
        M_5 = np.array([-0.03153728,0.01160465,1.00256374,0.99654512,0.10814223,0.03088474,-0.10843654,0.99619989,-0.01422552])
        M_6 = np.array([-0.00294161,-0.01878043,-0.99875921,-0.99913433,-0.00545105,0.00357126,-0.004815,0.99911992,-0.01849102])
        M_7 = np.array([-0.01694624,0.00893438,-1.00254205,-1.00234939,-0.00859451,0.01676781,-0.00850761,1.0027674,0.00960575])
        M_8 = np.array([-0.01233755,-0.00211036,-1.00689605,-1.00708674,-0.02531075,0.01233301,-0.02576082,1.00706965,-0.00212613])


        M_A = [M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8]
        
        for i, M in enumerate(M_A):
            M_A[i] = M.reshape((3, 3))
            
        return M_A
            
    else:
        M_10 = np.array([0.00529863,-0.00657411,-0.99965402,0.92140194,-0.38605527,0.00704625,-0.38633163,-0.92200509,0.0042057])
        M_9 = np.array([-0.05060716,-1.00568091,-0.03885759,-1.00772239,0.05060271,0.0010825,0.0006611,0.04126364,-1.0028714])
        M_11 = np.array([0.09562948,-0.99808126,-0.00550316,-0.0083807,0.00490206,-1.00708301,0.99761069,0.10039216,-0.00864757])
        M_12 = np.array([-5.40867212,1.13925683,-3.46278696,-0.72430491,0.7475252,4.3949523,1.28427441,7.06375231,-0.4813982])

        M_B = [M_9,M_10,M_11,M_12]
        
        for i, M in enumerate(M_B):
            M_B[i] = M.reshape((3, 3))
            
        return M_B
    
    
def rotate_24(soloA_bool):
    if soloA_bool:
        M_1 = np.array([-0.01151451,-0.03379413,-0.99912329,-0.99966879,0.02182397,0.01062976,0.02152466,0.99893068,-0.03362521])
        M_2 = np.array([0.0221395,-0.00814719,-1.00104386,-0.99984468,0.04330073,-0.02333449,0.04405573,1.00084827,-0.00708282])
        M_3 = np.array([0.03283952,-0.00829181,-0.99809581,-0.99786078,-0.01761458,-0.03303076,-0.01707146,0.99865858,-0.00872099])
        M_4 = np.array([0.00742014,-0.00079088,-1.00090218,-1.00039189,0.02286711,-0.00760828,0.02237911,1.00026345,-0.00006682])
        M_5 = np.array([-0.03153728,0.01160465,1.00256374,0.99654512,0.10814223,0.03088474,-0.10843654,0.99619989,-0.01422552])
        M_6 = np.array([-0.00294161,-0.01878043,-0.99875921,-0.99913433,-0.00545105,0.00357126,-0.004815,0.99911992,-0.01849102])
        M_7 = np.array([-0.01694624,0.00893438,-1.00254205,-1.00234939,-0.00859451,0.01676781,-0.00850761,1.0027674,0.00960575])
        M_8 = np.array([-0.01233755,-0.00211036,-1.00689605,-1.00708674,-0.02531075,0.01233301,-0.02576082,1.00706965,-0.00212613])



        M_A = [M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8]
        
        for i, M in enumerate(M_A):
            M_A[i] = M.reshape((3, 3))
            
        return M_A
            
    else:
        M_10 = np.array([0.00529863,-0.00657411,-0.99965402,0.92140194,-0.38605527,0.00704625,-0.38633163,-0.92200509,0.0042057])
        M_9 = np.array([-0.05060716,-1.00568091,-0.03885759,-1.00772239,0.05060271,0.0010825,0.0006611,0.04126364,-1.0028714])
        M_11 = np.array([0.09562948,-0.99808126,-0.00550316,-0.0083807,0.00490206,-1.00708301,0.99761069,0.10039216,-0.00864757])
        M_12 = np.array([-5.40867212,1.13925683,-3.46278696,-0.72430491,0.7475252,4.3949523,1.28427441,7.06375231,-0.4813982])


        M_B = [M_9,M_10,M_11,M_12]
        
        for i, M in enumerate(M_B):
            M_B[i] = M.reshape((3, 3))
            
        return M_B
            





rotate_21(True)

