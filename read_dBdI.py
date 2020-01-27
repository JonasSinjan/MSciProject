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
                #a = 1
                path_fol = f'C:\\Users\\jonas\\MSci-Code\\MSciProject\\dBdI_data\\Day{day}\\{sample}Hz\\{inst}_probe{probe}_vect_dict_{sample}Hz.csv'
            else:
                path_fol = os.path.expanduser(f"~/Documents/MSciProject/NewCode/dBdI_data/Day{day}/{sample}Hz/{inst}_probe{probe}_vect_dict_{sample}Hz.csv")


            df = pd.read_csv(path_fol)

            xdata = df['dI']
            probe_x_tmp = df['dB_X']
            probe_x_tmp_err = df['dB_X_err']
            probe_y_tmp = df['dB_Y']
            probe_y_tmp_err = df['dB_Y_err']
            probe_z_tmp = df['dB_Z']
            probe_z_tmp_err = df['dB_Z_err']

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
day = 2
instruments = ['EUI']
probes = [9]
sample_rate = 1

plot(windows,day,instruments,probes,sample_rate)

