import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv
import os

def plot_b_var_est(file_path, inst, plot = False):
    df = pd.read_csv(file_path)
    df = df.iloc[:-2] #excluding current and mag
    df['B_tot'] = np.sqrt(df['var_X']**2 + df['var_Y']**2 + df['var_Z']**2)
    df['B_tot_err'] = (1/df['B_tot']) * np.sqrt((df['var_X']*df['var_X_err'])**2 + (df['var_Y']*df['var_Y_err'])**2 + (df['var_Z']*df['var_Z_err'])**2)
    
    B_tot = list(df['B_tot'])
    B_tot_err = list(df['B_tot_err'])

    print(df.head())
    print(df.tail())

    #df['dist'] = [3.1622776601683795, 3.7721081638786553, 3.12, 3.4705619141574178, 3.1622776601683795, 1.8880677953929514, 1.12, 2.3976655313033137, 2.32800880582527, 4.154843559028427, 4.1089475538147235]
    
    if plot:
        
        ax = df.plot.bar(x='Probe', y = 'B_tot', yerr = 'B_tot_err', rot = 0, legend = False, title=f'{inst}_B_var')
        ax.set_ylabel("B_var [nT]")
        plt.show()
        """
        #plt.plot(dist,)
        ax = df.plot.bar(x='dist', y = 'B_tot', yerr = 'B_tot_err', rot = 0, legend = False, title=f'{inst}_B_var')
        ax.set_ylabel("B_var [nT]")
        plt.show()
        """
        

if __name__ == "__main__":
    windows = True
    inst = 'METIS'
    var = 1

    if windows:
        file_path = f'.\\Results\\variation\\{inst}_var{var}_B_variation_estimated.csv'
    else:
        file_path = os.path.expanduser(f"~/Documents/MSciProject/NewCode/Results/variation/{inst}_var{var}_B_variation_estimated.csv")

    plot_b_var_est(file_path, inst, plot = True)
    