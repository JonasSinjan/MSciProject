import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo
import numpy as np
from probe_dist import return_ypanel_dist, return_ypanel_loc

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

def get_csv_variation(inst, day, var_current, var_current_err):


    filename_MFSA = f'C:\\Users\\jonas\\MSci-Code\\MsciProject\\Results\\Gradient_dicts\\newdI_dicts\\Day_{day}\\cur\\{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv'
            
    grad_df = pd.read_csv(filename_MFSA)

    probe_list = return_ypanel_loc()
    distance,factor = return_ypanel_dist(probe_list)
    #B_variation = {}
    for j in range(11):
        probe = list(grad_df.iloc[j])
        dist = distance[j]
        if j == 8:
            j = 9
            probe = list(grad_df.iloc[9])
            dist = distance[9]
        elif j == 9:
            j = 8
            probe = list(grad_df.iloc[8])
            dist = distance[8]

        

        X_Slope,Y_Slope,Z_Slope,X_Slope_err,Y_Slope_err,Z_Slope_err = probe[1],probe[2],probe[3],probe[4],probe[5],probe[6]
        #print(X_Slope_err, Y_Slope_err, Z_Slope_err)
        #assuming intercept is zero
        B_var_X_estim = X_Slope * var_current
        B_var_Y_estim = Y_Slope * var_current
        B_var_Z_estim = Z_Slope * var_current


        B_var_X_err_estim = np.sqrt((X_Slope_err*var_current)**2 + (X_Slope*var_current_err)**2)
        B_var_Y_err_estim = np.sqrt((Y_Slope_err*var_current)**2 + (Y_Slope*var_current_err)**2)
        B_var_Z_err_estim = np.sqrt((Z_Slope_err*var_current)**2 + (Z_Slope*var_current_err)**2)

        
        #BETTER TO ALSO CALCULATE TOTAL B VAR AND ERR HERE IN FILE
        B_tot_var = np.sqrt(B_var_X_estim**2 + B_var_Y_estim**2 + B_var_Z_estim**2)

        B_tot_err = 1/B_tot_var * np.sqrt((B_var_X_err_estim*B_var_X_estim)**2 + (B_var_Y_err_estim*B_var_Y_estim)**2 +(B_var_Z_err_estim*B_var_Z_estim)**2)

        print(f'Probe {j+1}', round(dist,2), round(B_tot_var,2))
        
        #B_variation[f'{j+1}'] = [B_var_X_estim,B_var_Y_estim,B_var_Z_estim,B_var_X_err_estim,B_var_Y_err_estim,B_var_Z_err_estim]
        plt.figure(1)    
        plt.scatter(dist, B_tot_var)
        plt.errorbar(dist, B_tot_var, yerr = B_tot_err, linestyle = "None")

    plt.show()
            
    

if __name__ == "__main__":
    
    #filepath = '.\\Results\\eui_signals_day2.csv'

    #plot_signals(filepath, 0.8)

    day = 2
    inst = 'EUI'
    var_current = 0.8
    var_current_err = 0.05

    get_csv_variation(inst, day, var_current, var_current_err)