import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv
import os
import scipy.optimize as spo

def plot_b_var_est(file_path, inst, day, plot = False):
    df = pd.read_csv(file_path)
    df = df.iloc[:-3] #excluding current and mag
    df['B_tot'] = np.sqrt(df['var_X']**2 + df['var_Y']**2 + df['var_Z']**2)
    df['B_tot_err'] = (1/df['B_tot']) * np.sqrt((df['var_X']*df['var_X_err'])**2 + (df['var_Y']*df['var_Y_err'])**2 + (df['var_Z']*df['var_Z_err'])**2)
    
    B_tot = list(df['B_tot'])
    B_tot_err = list(df['B_tot_err'])

    #print(df.head())
    #print(df.tail())
    dist_list = [3.1622776601683795, 3.7721081638786553, 3.12, 3.4705619141574178, 3.1622776601683795, 1.8880677953929514, 1.12, 2.3976655313033137, 2.32800880582527, 4.154843559028427, 4.1089475538147235]#, 12.681225256259744]
    
    print(df.index.size, len(dist_list))
    df['dist'] = [round(i,3) for i in dist_list]
    
    if plot:
        
        ax = df.plot.bar(x='Probe', y = 'B_tot', yerr = 'B_tot_err', rot = 0, legend = False, title=f'{inst}_B_var')
        ax.set_ylabel("B_var [nT]")
        ax.set_title(f'{inst} B_var estimates - Day {day}')
        plt.show()
        """
        ax = df.plot.bar(x='dist', y = 'B_tot', yerr = 'B_tot_err', rot = 0, legend = False, title=f'{inst}_B_var')
        ax.set_ylabel("B_var [nT]")
        plt.show()
        """

        def cubic(x,a,b):
                return a*(x**(-3)) + b
        df = df.sort_values('dist', ascending = True, kind = 'mergesort')
        params,cov = spo.curve_fit(cubic, df['dist'], df['B_tot'], sigma = df['B_tot_err'], absolute_sigma = True)
        perr = np.sqrt(np.diag(cov))

        plt.figure()
        plt.scatter(df['dist'],df['B_tot'])
        plt.errorbar(df['dist'],df['B_tot'],yerr=df['B_tot_err'], linestyle="None")
        xdata = np.linspace(1,4.5,100)
        plt.plot(xdata, params[0]*(xdata)**(-3) + params[1], 'b-',label= f'y = {params[0]}/r^3 + {params[1]} ')
        plt.xlabel('r (distance from centre of -Y instrument panel) [m]')
        plt.ylabel('B_var [nT]')
        plt.title(f'1/r^3 Dipole Fit -  {inst} - Day {day}')
        print(params, perr)
        plt.show()


if __name__ == "__main__":
    windows = False
    inst = 'METIS'
    var = 1
    day =  1
    if windows:
        file_path = f'.\\Results\\variation\\{inst}_var{var}_B_variation_estimated_day{day}.csv'
    else:
        file_path = os.path.expanduser(f"~/Documents/MSciProject/NewCode/Results/variation/{inst}_var{var}_B_variation_estimated.csv")

    plot_b_var_est(file_path, inst, day, plot = True)
    