import matplotlib.pyplot as plt
import matplotlib as mpl
from processing import processing
from pandas.plotting import register_matplotlib_converters
from current import current_peaks
register_matplotlib_converters()
import scipy.stats as spstats
import pandas as pd
import os
import numpy as np
import csv 



def plot(windows,day,instruments,probes,sample_rate):
    for inst in instruments:
        for probe in probes:

            if sample_rate == 1000:
                sample = '1k'
            elif sample_rate == 1:
                sample = '1'

            if windows:
                path = f'C:\\Users\\jonas\\MSci-Code\\MSciProject\\dBdI_data\\Day{day}\\{sample}Hz\\{inst}\\{inst}_probe{probe}_vect_dict_{sample}Hz.csv'
                if day == 1:
                    path = f'C:\\Users\\jonas\\MSci-Code\\MSciProject\\dBdI_data\\Day{day}\\{sample}Hz\\{inst}\\{inst}_probe{probe}_vect_dict_{sample}Hz_day1.csv'
                    err_path = f'C:\\Users\\jonas\\MSci-Code\\MSciProject\\day1_mfsa_probe_vars.csv'
                elif day == 2:
                    err_path = f'C:\\Users\\jonas\\MSci-Code\\MSciProject\\day2_mfsa_probe_vars.csv'
            else:
                path = os.path.expanduser(f"~/Documents/MSciProject/NewCode/dBdI_data/Day{day}/{sample}Hz/{inst}/{inst}_probe{probe}_vect_dict_{sample}Hz.csv")
                if day == 1:
                    path = os.path.expanduser(f"~/Documents/MSciProject/NewCode/dBdI_data/Day{day}/{sample}Hz/{inst}/{inst}_probe{probe}_vect_dict_{sample}Hz_day1.csv")
                    err_path = os.path.expanduser(f"~/Documents/MSciProject/NewCode/day1_mfsa_probe_vars.csv")
                elif day == 2:
                    err_path = os.path.expanduser(f"~/Documents/MSciProject/NewCode/day2_mfsa_probe_vars.csv")

            df = pd.read_csv(path)
            df_err_correction = pd.read_csv(err_path)
            print(df_err_correction.head())#['Bx_var'])
            df_err = df_err_correction.iloc[probe-1]

            xdata = df['dI']
            probe_x_tmp = df['dB_X']
            probe_x_tmp_err = np.sqrt(df['dB_X_err']**2 + df_err['Bx_var']**2)
            probe_y_tmp = df['dB_Y']
            probe_y_tmp_err = np.sqrt(df['dB_Y_err']**2 + df_err['By_var']**2)
            probe_z_tmp = df['dB_Z']
            probe_z_tmp_err = np.sqrt(df['dB_Z_err']**2 + df_err['Bz_var']**2)

            X = spstats.linregress(xdata, probe_x_tmp) #adding bonus point has little effect on grad - only changes intercept
            Y = spstats.linregress(xdata, probe_y_tmp)
            Z = spstats.linregress(xdata, probe_z_tmp)

            print('sps.linregress')
            print('Slope = ', X.slope, '+/-', X.stderr, ' Intercept = ', X.intercept)
            print('Slope = ', Y.slope, '+/-', Y.stderr, ' Intercept = ', Y.intercept)
            print('Slope = ', Z.slope, '+/-', Z.stderr, ' Intercept = ', Z.intercept)

            plt.figure()
            plt.errorbar(xdata, probe_x_tmp, yerr = probe_x_tmp_err, fmt = 'bs',label = f'X grad: {round(X.slope,2)} ± {round(X.stderr,2)} int: {round(X.intercept, 2)}', markeredgewidth = 2)
            plt.errorbar(xdata, probe_y_tmp, yerr = probe_y_tmp_err, fmt = 'rs', label = f'Y grad: {round(Y.slope,2)} ± {round(Y.stderr,2)} int: {round(Y.intercept, 2)}', markeredgewidth = 2)
            plt.errorbar(xdata, probe_z_tmp, yerr = probe_z_tmp_err, fmt = 'gs', label = f'Z grad: {round(Z.slope,2)} ± {round(Z.stderr,2)} int: {round(Z.intercept, 2)}', markeredgewidth = 2)

            plt.plot(xdata, X.intercept + X.slope*np.array(xdata), 'b-')
            plt.plot(xdata, Y.intercept + Y.slope*np.array(xdata), 'r-')
            plt.plot(xdata, Z.intercept + Z.slope*np.array(xdata), 'g-')

            plt.legend(loc="best")
            plt.title(f'Day {day} - {inst} - Probe {probe} - {sample}Hz - MFSA')
            plt.xlabel('dI [A]')
            plt.ylabel('dB [nT]')
            

            plt.show()
        


windows = True
day = 1
instruments = ['EUI']
probes = [3,9,10,11]
sample_rate = 1

plot(windows,day,instruments,probes,sample_rate)

