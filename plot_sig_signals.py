import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo
import numpy as np

def plot_signals(filepath, max_current):

    df = pd.read_csv(filepath)
    #print(df.head())
    #print(df.columns.tolist())
    x = df['distance']
    y = np.sqrt((df['grad_x'])**2 + (df['grad_y'])**2 + (df['grad_z'])**2)*max_current
    yerr = np.sqrt((df['grad_x']*max_current/y)**2*df['grad_x_err']**2 + (df['grad_y']*max_current/y)**2*df['grad_y_err']**2 + (df['grad_z']*max_current/y)**2*df['grad_z_err']**2)
    plt.figure()
    plt.scatter(x,y, label='_nolegend_')
    plt.errorbar(x,y, yerr = yerr, linestyle="None", label='_nolegend_')
    plt.show()

if __name__ == "__main__":
    
    filepath = '.\\Results\\eui_signals_day2.csv'

    plot_signals(filepath, 0.8)