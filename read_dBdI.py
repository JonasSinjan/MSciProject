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
import scipy.optimize as spo



def plot_old_errs_with_lin(windows,day,instruments,probes,sample_rate):
    for inst in instruments:
        for probe in probes:

            if sample_rate == 1000:
                sample = '1k'
            elif sample_rate == 1:
                sample = '1'

            if windows:
                path = f'.\\Results\\dBdI_data\\Day{day}\\{sample}Hz\\{inst}\\{inst}_probe{probe}_vect_dict_{sample}Hz.csv'
                if day == 1:
                    path = f'.\\Results\\dBdI_data\\Day{day}\\{sample}Hz\\{inst}\\{inst}_probe{probe}_vect_dict_{sample}Hz_day1.csv'
                    err_path = f'.\\day1_mfsa_probe_vars.csv'
                elif day == 2:
                    err_path = f'.\\day2_mfsa_probe_vars.csv'
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

def plot_new_curve(windows, day, instruments, probes, sample):
    for inst in instruments:
        for probe in probes:

            if windows:
                path = f'.\\Results\\dBdI_data\\Day{day}\\{sample}Hz_with_err\\{inst}\\{inst}_probe{probe}_vect_dict_{sample}Hz_day{day}.csv'

            else:
                path = os.path.expanduser(f"~/Documents/MSciProject/NewCode/dBdI_data/Day{day}/{sample}Hz_with_err/{inst}/{inst}_probe{probe}_vect_dict_{sample}Hz_day{day}.csv")
        
            df = pd.read_csv(path)
            xdata = df['dI']
            probe_x_tmp = df['dB_X']
            probe_y_tmp = df['dB_Y']
            probe_z_tmp = df['dB_Z']

            probe_x_tmp_err = df['dB_X_err']
            probe_y_tmp_err = df['dB_Y_err']
            probe_z_tmp_err = df['dB_Z_err']

            def line(x,a,b):
                return a*x + b

            params_x,cov_x = spo.curve_fit(line, xdata, probe_x_tmp[:], sigma = probe_x_tmp_err[:], absolute_sigma = True)
            params_y,cov_y = spo.curve_fit(line, xdata, probe_y_tmp[:], sigma = probe_y_tmp_err[:], absolute_sigma = True)
            params_z,cov_z = spo.curve_fit(line, xdata, probe_z_tmp[:], sigma = probe_z_tmp_err[:], absolute_sigma = True)

            perr_x = np.sqrt(np.diag(cov_x))
            perr_y = np.sqrt(np.diag(cov_y))
            perr_z = np.sqrt(np.diag(cov_z))
            
            print('spo.curve_fit')
            print('Slope = ', params_x[0], '+/-', perr_x[0], 'Intercept = ', params_x[1], '+/-', perr_x[1])
            print('Slope = ', params_y[0], '+/-', perr_y[0], 'Intercept = ', params_y[1], '+/-', perr_y[1])
            print('Slope = ', params_z[0], '+/-', perr_z[0], 'Intercept = ', params_z[1], '+/-', perr_z[1])

            plt.plot(xdata, params_x[0]*xdata + params_x[1], 'b-',label='_nolegend_')
            plt.plot(xdata, params_y[0]*xdata + params_y[1], 'r-',label='_nolegend_')
            plt.plot(xdata, params_z[0]*xdata + params_z[1], 'g-',label='_nolegend_')

            plt.errorbar(xdata, probe_x_tmp, yerr = probe_x_tmp_err, fmt = 'bs', markeredgewidth = 2, label = f'curve_fit - X grad: {round(params_x[0],2)} ± {round(perr_x[0],2)} int: {round(params_x[1],2)} ± {round(perr_x[1],2)}')
            plt.errorbar(xdata, probe_y_tmp, yerr = probe_y_tmp_err, fmt = 'rs', markeredgewidth = 2, label = f'curve_fit - Y grad: {round(params_y[0],2)} ± {round(perr_y[0],2)} int: {round(params_y[1],2)} ± {round(perr_y[1],2)}')
            plt.errorbar(xdata, probe_z_tmp, yerr = probe_z_tmp_err, fmt = 'gs', markeredgewidth = 2, label = f'curve_fit - Z grad: {round(params_z[0],2)} ± {round(perr_z[0],2)} int: {round(params_z[1],2)} ± {round(perr_z[1],2)}')

            plt.legend(loc="best")
            plt.title(f'Day {day} - {inst} - Probe {probe} - {sample}Hz - MFSA')
            plt.xlabel('dI [A]')
            plt.ylabel('dB [nT]')
            plt.show()

if __name__ == "__main__":
    windows = True
    day = 1
    instruments = ['METIS']#['EUI', 'METIS', 'PHI', 'SWA', 'EPD', 'SoloHI', 'STIX', 'SPICE']
    probes = range(1,13)
    sample_rate = 1

    #plot_old_errs_with_lin(windows,day,instruments,probes,sample_rate)

    plot_new_curve(windows, day, instruments, probes, sample_rate)

