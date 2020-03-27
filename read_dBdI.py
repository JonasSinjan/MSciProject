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
import seaborn as sns
#print(mpl.font_manager.get_cacheddir())
#mpl.rcParams['font.family'] = ['sans-serif']
#mpl.rcParams['font.sans-serif'] = ['cmr10']
#mpl.rcParams['text.usetex'] = True


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
            plt.errorbar(xdata, probe_x_tmp, yerr = probe_x_tmp_err, fmt = 'bs',label = f'X grad: {round(X.slope,2)} $\pm$ {round(X.stderr,2)} int: {round(X.intercept, 2)}', markeredgewidth = 2)
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
        vect_dict = {}
        for probe in probes:

            if windows:
                #path = f'.\\Results\\dBdI_data\\old_dI\\Day{day}\\{sample}Hz_with_err\\{inst}\\{inst}_probe{probe}_vect_dict_{sample}Hz_day{day}.csv'
                path = f'.\\Results\\dBdI_data\\new_dI_copy\\Day{day}\\{inst}\\{inst}_probe{probe}_vect_dict_{sample}Hz_day{day}.csv'

            else:
                path = os.path.expanduser(f"~/Documents/MSciProject/NewCode//Results/dBdI_data/Day{day}/{sample}Hz_with_err/{inst}/{inst}_probe{probe}_vect_dict_{sample}Hz_day{day}.csv")
        
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
            print('Slope = ', round(params_x[0],2), '+/-', round(perr_x[0],2))#, 'Intercept = ', params_x[1], '+/-', perr_x[1])
            print('Slope = ', round(params_y[0],2), '+/-', round(perr_y[0],2))#, 'Intercept = ', params_y[1], '+/-', perr_y[1])
            print('Slope = ', round(params_z[0],2), '+/-', round(perr_z[0],2))#, 'Intercept = ', params_z[1], '+/-', perr_z[1])
            #g = plt.figure()
            plt.plot(xdata, params_x[0]*xdata + params_x[1], '-', color = u'#1f77b4', label='_nolegend_')
            plt.plot(xdata, params_y[0]*xdata + params_y[1], '-', color = u'#ff7f0e', label='_nolegend_')
            plt.plot(xdata, params_z[0]*xdata + params_z[1], '-', color = u'#2ca02c', label='_nolegend_')

            cap_size = 3
            error_colour = None
            eline_width = 1.5
            cap_thick = 0
            ms = 8
            #u'#1f77b4', u'#ff7f0e', u'#2ca02c'
            plt.errorbar(xdata, probe_x_tmp, yerr = probe_x_tmp_err, fmt = 'o', color = u'#1f77b4', capsize = cap_size, capthick = cap_thick, ecolor = error_colour, elinewidth = eline_width, markeredgecolor = "white", markersize = ms, label = f'X grad: {round(params_x[0],2)} $\pm$ {round(perr_x[0],2)} nT/A')#, int: {round(params_x[1],2)} ± {round(perr_x[1],2)}')
            plt.errorbar(xdata, probe_y_tmp, yerr = probe_y_tmp_err, fmt = 'o', color=u'#ff7f0e', capsize = cap_size, capthick = cap_thick, ecolor = error_colour, elinewidth = eline_width, markeredgecolor = "white", markersize = ms, label = f'Y grad: {round(params_y[0],2)} ± {round(perr_y[0],2)} nT/A')#', int: {round(params_y[1],2)} ± {round(perr_y[1],2)}')
            plt.errorbar(xdata, probe_z_tmp, yerr = probe_z_tmp_err, fmt = 'o', color = u'#2ca02c', capsize = cap_size, capthick = cap_thick, ecolor = error_colour, elinewidth = eline_width, markeredgecolor = "white", markersize = ms, label = f'Z grad: {round(params_z[0],2)} ± {round(perr_z[0],2)} nT/A')#', int: {round(params_z[1],2)} ± {round(perr_z[1],2)}')
            
            #sns.pointplot(data = [xdata, probe_x_tmp], fmt = 'bs', label = f'X grad: {round(params_x[0],2)} ± {round(perr_x[0],2)} nT/A')#, int: {round(params_x[1],2)} ± {round(perr_x[1],2)}')
            #sns.pointplot(data = [xdata, probe_y_tmp], fmt = 'bs', label = f'X grad: {round(params_y[0],2)} ± {round(perr_y[0],2)} nT/A')#', int: {round(params_y[1],2)} ± {round(perr_y[1],2)}')
            #sns.pointplot(data = [xdata, probe_z_tmp], fmt = 'bs', label = f'X grad: {round(params_z[0],2)} ± {round(perr_z[0],2)} nT/A')#', int: {round(params_z[1],2)} ± {round(perr_z[1],2)}')
            
            #d = {'xdata': xdata, 'probe_x_tmp': probe_x_tmp, 'probe_x_tmp_err':probe_x_tmp_err}
            #df = pd.DataFrame(data = d)
            #g = sns.FacetGrid(df, size = 5)
            #g.map(plt.errorbar, "xdata", "probe_x_tmp", "probe_x_tmp_err", fmt = 'bs')

            plt.legend(loc="best")
            plt.title(f'Day {day} - {inst} - Probe {probe} - {sample}Hz')
            plt.xlabel('dI [A]')
            plt.ylabel('dB [nT]')
            plt.show()
            
            #   this """ section is to update the gradient dicts using the corrected dbdi data
            
            vect_dict[f'{probe}'] = [params_x[0], params_y[0], params_z[0], perr_x[0], perr_y[0], perr_z[0], params_x[1], params_y[1], params_z[1], perr_x[1], perr_y[1], perr_z[1]]
            #print(vect_dict[f'{probe}'])
        """
        w = csv.writer(open(f".\\Results\\Gradient_dicts\\100pT_error_dicts\\Day_{day}\\cur\\{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv", "w"))
        w.writerow(["Probe","X.slope_cur", "Y.slope_cur", "Z.slope_cur","X.slope_cur_err", "Y.slope_cur_err", "Z.slope_cur_err","X_zero_int","Y_zero_int","Z_zero_int", "X_zero_int_err","Y_zero_int_err","Z_zero_int_err"])
        for key, val in vect_dict.items():
            w.writerow([key,val[0],val[1],val[2],val[3],val[4],val[5],val[6],val[7],val[8],val[9],val[10],val[11]])
        """
        
if __name__ == "__main__":
    windows = True
    day = 2
    instruments = ['EUI']#['EUI', 'METIS', 'PHI', 'SWA', 'SoloHI', 'STIX', 'SPICE', 'EPD']
    probes = [7]
    sample_rate = 1

    #plot_old_errs_with_lin(windows,day,instruments,probes,sample_rate)
    
    plot_new_curve(windows, day, instruments, probes, sample_rate)
