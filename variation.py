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
from mag import mag
import csv
from processing import processing

windows = False
day = 2
inst = 'METIS'


####  CALCULATING dB USING CURRENT ####

if windows:
    filename_current = f'C:\\Users\\jonas\\MSci-Data\\LCL_data\\Day {day} Payload LCL Current Profiles.xlsx'
    filename_mag = f'C:\\Users\\jonas\\MSci-Data\\mag\\PoweredDay2.csv.txt'
else: 
    filename_current = os.path.expanduser(f"~/Documents/MSciProject/Data/LCL_data/Day_{day}_Payload_LCL_Current_Profiles.xlsx")
    filename_mag = os.path.expanduser(f"~/Documents/MSciProject/Data/mag/PoweredDay2.csv.txt")

df_current =  pd.read_excel(filename_current)
df_mag = mag(filename_mag, day, start_dt=None, end_dt=None, plot = False)
df_current.set_index(['EGSE Time'], inplace = True)

#df = df.resample(f'{1}s').mean()
#eui 9:24 - 10:10
#eui '9:40:30', '10:10')
#phi 8:03 - 8:40
#metis 10:10 - 10:57

df_current = df_current[f'{inst} Current [A]']

df_current_list = []
df_mag_list = []
var_current = []
var_current_err = []
var_mag_X = []
var_mag_X_err = []
var_mag_Y = []
var_mag_Y_err = []
var_mag_Z = []
var_mag_Z_err = []

if inst == 'EUI':
    df_current_list.append(df_current.between_time('9:49:00', '9:54:40'))
    df_current_list.append(df_current.between_time('9:54:30', '10:03:20'))
    df_mag_list.append(df_mag.between_time('9:49:00', '9:54:40'))
    df_mag_list.append(df_mag.between_time('9:54:30', '10:03:20'))
elif inst == 'METIS':
    df_current_list.append(df_current.between_time('10:35:30', '10:55'))
    df_mag_list.append(df_mag.between_time('10:35:30', '10:55'))


for df in df_current_list:
    plt.figure()
    plt.plot(df.index.time, df)
    plt.xlabel('Time [H:M:S]')
    plt.ylabel('Current [A]')
    plt.title(f'{inst} CURRENT PROFILE')
    plt.show()

    #calculate amplition of variation
    mean_val = df.mean()
    df_top = df[df > mean_val]
    df_bot = df[df < mean_val]
    top_avg = df_top.mean()
    top_std = df_top.std()/np.sqrt(len(df_top))
    bot_avg = df_bot.mean()
    bot_std = df_bot.std()/np.sqrt(len(df_bot))
    tot_std = np.sqrt(bot_std**2 + top_std**2)
    dif = top_avg - bot_avg

    var_current.append(dif)
    var_current_err.append(tot_std)

    #print("current variation (A)", dif,"+/-" , tot_std)



for df in df_mag_list:
    plt.figure()
    plt.plot(df.index.time, df)
    plt.xlabel('Time [H:M:S]')
    plt.ylabel('mag [T]')
    plt.title(f'{inst} MAG PROFILE')
    plt.show()

    col_list = ['X','Y','Z']

    for col in col_list:
        #calculate amplition of variation
        mean_val = df[col].mean()
        df_top = df[col][df[col] > mean_val]
        df_bot = df[col][df[col] < mean_val]
        top_avg = df_top.mean()
        top_std = df_top.std()/np.sqrt(len(df_top))
        bot_avg = df_bot.mean()
        bot_std = df_bot.std()/np.sqrt(len(df_bot))
        tot_std = np.sqrt(bot_std**2 + top_std**2)
        dif = top_avg - bot_avg

        if col == 'X':
            var_mag_X.append(dif)
            var_mag_X_err.append(tot_std)
        if col == 'Y':
            var_mag_Y.append(dif)
            var_mag_Y_err.append(tot_std)
        if col == 'Z':
            var_mag_Z.append(dif)
            var_mag_Z_err.append(tot_std)

    #print("mag variation (T)", dif,"+/-" , tot_std)


if windows: #jonas put ur equivalent filepath here 
    filename_MFSA = f'C:\\Users\\jonas\\MSci-Data\\Gradient_dicts\\Day_{day}\\1Hz_NoOrigin\\{inst}_vect_dict_NOORIGIN.csv")'
else: 
    filename_MFSA = os.path.expanduser(f"~/Documents/MSciProject/NewCode/Gradient_dicts/Day_{day}/1Hz_NoOrigin/{inst}_vect_dict_NOORIGIN.csv")
        
grad_df = pd.read_csv(filename_MFSA)

B_variation = {}

for i in range(len(var_current)):
    for j in range(11):
        
        probe = list(grad_df.iloc[j])

        X_Slope,Y_Slope,Z_Slope,X_Slope_err,Y_Slope_err,Z_Slope_err = probe[1],probe[2],probe[3],probe[4],probe[5],probe[6]

        #assuming intercept is zero
        B_var_X_estim = X_Slope * var_current[i]
        B_var_Y_estim = Y_Slope * var_current[i]
        B_var_Z_estim = Z_Slope * var_current[i]

        B_var_X_err_estim = np.sqrt(X_Slope_err**2 + var_current_err[i]**2)
        B_var_Y_err_estim = np.sqrt(Y_Slope_err**2 + var_current_err[i]**2)
        B_var_Z_err_estim = np.sqrt(Z_Slope_err**2 + var_current_err[i]**2)

        #print (f"Extimated B variation, probe {j} X:", B_var_X_estim, "+/-", B_var_X_err_estim)
        #print (f"Extimated B variation, probe {j} Y:", B_var_Y_estim, "+/-", B_var_Y_err_estim)
        #print (f"Extimated B variation, probe {j} Z:", B_var_Z_estim, "+/-", B_var_Z_err_estim)
        
        B_variation[f'{j+1}'] = [B_var_X_estim,B_var_Y_estim,B_var_Z_estim,B_var_X_err_estim,B_var_Y_err_estim,B_var_Z_err_estim]
    
    
    #print (f"mag B variation X:", var_mag_X[i], "+/-", var_mag_X_err[i])
    #print (f"mag B variation Z:", var_mag_Z[i], "+/-", var_mag_Z_err[i])
    #print (f"mag B variation Y:", var_mag_Y[i], "+/-", var_mag_Y_err[i])
    
    B_variation['mag'] = [var_mag_X[i],var_mag_Y[i],var_mag_Z[i],var_mag_X_err[i],var_mag_Y_err[i],var_mag_Z_err[i]]
    B_variation['current'] = [var_current[i],var_current[i],var_current[i],var_current_err[i],var_current_err[i],var_current_err[i]]
    

    w = csv.writer(open(f"{inst}_var{i+1}_B_variation_estimated.csv", "w"))
    w.writerow(["Probe","var_X", "var_Y", "var_Z","var_X_err", "var_Y_err", "var_Z_err"])
    for key, val in B_variation.items():
        w.writerow([key,val[0],val[1],val[2],val[3],val[4],val[5]])
    

"""

####  CALCULATING dB FOR PROBES - UNFINISHED  ####

if day == 1:
    if windows:
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_one\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_one\B'
    else:
        path_fol_A = os.path.expanduser("~/Documents/MSciProject/Data/day_one/A")
        path_fol_B = os.path.expanduser("~/Documents/MSciProject/Data/day_one/B")
elif day == 2:
    if windows:
        path_fol_A = r'C:\Users\jonas\MSci-Data\day_two\A'
        path_fol_B = r'C:\Users\jonas\MSci-Data\day_two\B'
    else:
        path_fol_A = os.path.expanduser("~/Documents/MSciProject/Data/day_two/A")
        path_fol_B = os.path.expanduser("~/Documents/MSciProject/Data/day_two/B")



start_times = []
end_times = []
sampling_freq = 1000

if inst == 'EUI':
    start_times.append('9:49:0','9:54:30')
    end_times.append('9:54:40','10:03:20')
elif inst == 'METIS':
    start_times.append('10:35:30')
    end_times.append('10:55:0')

for time in start_times:
    start_times[start_times.index(time)] = datetime.strptime(time, '%H:%M:%S')
for time in end_times:
    end_times[end_times.index(time)] = datetime.strptime(time, '%H:%M:%S')


var_MFSA_X = []
var_MFSA_Y = []
var_MFSA_Z = []
var_MFSA_X_err = []
var_MFSA_Y_err = []
var_MFSA_Z_err = []

for i in range(len(start_times)):

    start_csv_A, end_csv_A = processing.which_csvs(True, day ,start_times[0], end_times[0], tz_MAG = True)
    start_csv_B, end_csv_B = processing.which_csvs(False, day ,start_times[0], end_times[0], tz_MAG = True)

    all_files_A = [0]*(end_csv_A + 1 - start_csv_A)

    if day == 1:
        for index, j in enumerate(range(start_csv_A, end_csv_A + 1)): #this will loop through and add the csv files that contain the start and end time set above
            if windows:
                all_files_A[index] = path_fol_A + f'\SoloA_2019-06-21--08-10-10_{j}.csv'
            else:
                all_files_A[index] = path_fol_A + os.path.expanduser(f'/SoloA_2019-06-21--08-10-10_{j}.csv') #need to change path_fol_A  to the path where your A folder is
        
        all_files_B = [0]*(end_csv_B + 1 - start_csv_B)
        for index, j in enumerate(range(start_csv_B, end_csv_B + 1)): 
            if windows:
                all_files_B[index] = path_fol_B + f'\SoloB_2019-06-21--08-09-10_{j}.csv'
            else:
                all_files_B[index] = path_fol_B + os.path.expanduser(f'/SoloB_2019-06-21--08-09-10_{j}.csv') #need to change path_f

    if day == 2:
        for index, j in enumerate(range(start_csv_A, end_csv_A + 1)): #this will loop through and add the csv files that contain the start and end time set above
            if windows:
                all_files_A[index] = path_fol_A + f'\SoloA_2019-06-24--08-14-46_{j}.csv'
            else:
                all_files_A[index] = path_fol_A + os.path.expanduser(f'/SoloA_2019-06-24--08-14-46_{j}.csv') #need to change path_fol_A  to the path where your A folder is
        
        all_files_B = [0]*(end_csv_B + 1 - start_csv_B)
        for index, j in enumerate(range(start_csv_B, end_csv_B + 1)): 
            if windows:
                all_files_B[index] = path_fol_B + f'\SoloB_2019-06-24--08-14-24_{j}.csv'
            else:
                all_files_B[index] = path_fol_B + os.path.expanduser(f'/SoloB_2019-06-24--08-14-24_{j}.csv') #need to change path_f



    for j in range(11):
        #looping through each sensor
        if j < 8:
            soloA_bool = True
            all_files = all_files_A
        else:
            soloA_bool = False
            all_files = all_files_B
        if j < 9:
            num_str = f'0{i+1}'
        else: 
            num_str = j+1
        
        collist = ['time', f'Probe{num_str}_X', f'Probe{num_str}_Y', f'Probe{num_str}_Z']
            
        if soloA_bool:
            df = processing.read_files(all_files, soloA_bool, windows, sampling_freq, collist, day=day, start_dt = start_times[i], end_dt = end_times[i])
            rotate_mat = processing.rotate_24(soloA_bool)[j]
        else:
            df = processing.read_files(all_files, soloA_bool, windows, sampling_freq, collist, day=day, start_dt = start_times[i], end_dt = end_times[i])
            rotate_mat = processing.rotate_24(soloA_bool)[j-8]
        df.iloc[:,0:3] = np.matmul(rotate_mat, df.iloc[:,0:3].values.T).T
        
    
        df = processing.shifttime(df, soloA_bool) # must shift MFSA data to MAG/spacecraft time
        

        df = df.between_time(start_times[i], end_times[i])



        col_list = [f'Probe{num_str}_X',f'Probe{num_str}_Y',f'Probe{num_str}_Z']

        for col in col_list:
            #calculate amplition of variation
            mean_val = df[col].mean()
            df_top = df[col][df[col] > mean_val]
            df_bot = df[col][df[col] < mean_val]
            top_avg = df_top.mean()
            top_std = df_top.std()/np.sqrt(len(df_top))
            bot_avg = df_bot.mean()
            bot_std = df_bot.std()/np.sqrt(len(df_bot))
            tot_std = np.sqrt(bot_std**2 + top_std**2)
            dif = top_avg - bot_avg

            if col == f'Probe{num_str}_X':
                var_MFSA_X.append(dif)
                var_MFSA_X_err.append(tot_std)
            if col == f'Probe{num_str}_Y':
                var_MFSA_Y.append(dif)
                var_MFSA_Y_err.append(tot_std)
            if col == f'Probe{num_str}_Z':
                var_MFSA_Z.append(dif)
                var_MFSA_Z_err.append(tot_std)

        B_variation[f'{j+1}'] = [var_MFSA_X[i],var_MFSA_Y[i],var_MFSA_Z[i],var_MFSA_X_err[i],var_MFSA_Y_err[i],var_MFSA_Z_err[i]]
        w = csv.writer(open(f"{inst}_var{i+1}_B_variation_MFSA.csv", "w"))
        w.writerow(["Probe","var_X", "var_Y", "var_Z","var_X_err", "var_Y_err", "var_Z_err"])
        for key, val in B_variation.items():
            w.writerow([key,val[0],val[1],val[2],val[3],val[4],val[5]])

        #print("mag variation (T)", dif,"+/-" , tot_std)
"""

"""

# -------METIS-----------#
#metis current variation during scientific operation is 0.1286 +/- 0.0003 A

#metis only significant signal in Y at probe 10: 0.29nT/A
var = 0.29*dif
#when multiplying together, add the fractional errors in quadrature
err = np.sqrt((0.13/0.29)**2 + (tot_std/dif)**2) #fractional error in variation
print(round(var, 4), '+/-', round(err*var,4), 'nT') #in nT
#metis B var at MAG-OBS: 37 +/- 17 pT


# -------EUI----------#
dif = 0.8 #rough estimate as variation not as constant as with metis
tot_std = 0.1
tot_grad = np.sqrt(0.71**2 + 1.02**2)

print(tot_grad, 'nT/A')

grad_err = np.sqrt(((0.71/tot_grad)**2)*(0.19**2) + ((1.02/tot_grad)**2)*(0.19**2))

var = dif * tot_grad

var_err = np.sqrt((grad_err/var)**2 + (tot_std/dif)**2)
print(1000*round(var, 5), '+/-', 1000*round(var_err*var,5), 'pT') #in nT


"""