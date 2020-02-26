import numpy as np
import math
import pandas as pd
from probe_dist import return_ypanel_dist, return_ypanel_loc
import os
import matplotlib.pyplot as plt

def solver(file_path, inst, day, variation_est=True):
    df = pd.read_csv(file_path)
    if variation_est:
        df = df.iloc[:-3] #excluding current and mag
    else:
        df = df.iloc[:-1] #exclude probe 12

    probe_loc_list = return_ypanel_loc()
    probe_loc_list = probe_loc_list[:-1] #dont want probe 12 location

    probe_dist_list, factor = return_ypanel_dist(probe_loc_list)

    mx_list, my_list, mz_list = 11*[0],  11*[0], 11*[0] 
    for i in range(0,11):
        df_tmp = df.iloc[i]
        if variation_est:
            b = np.array([df_tmp['var_X'], df_tmp['var_Y'], df_tmp['var_Z']])
        else:
            b = np.array([df_tmp['X.slope_cur'], df_tmp['Y.slope_cur'], df_tmp['Z.slope_cur']])

        print('Probe ', i+1)
        loc = probe_loc_list[i]
        x, y, z = loc[0], loc[1], loc[2]
        print('x,y,z = ', x,y,z)
        r = probe_dist_list[i]
        print('r = ', r)

        
        a = np.array([[3*(x**2)/r**5-1/r**3, 3*y*z/r**5, 3*z*x/r**5],[3*x*y/r**5, 3*(y**2)/r**5-1/r**3, 3*z*y/r**5],[3*x*z/r**5, 3*y*z/r**5, 3*(z**2)/r**5-1/r**3]])
        a = 10**(-7)*a
        b = 10**-9*b #get in units of Tesla
        m = np.linalg.solve(a,b)

        print(m) #m in units of Amp*m**2
        mx_list[i] = m[0]
        my_list[i] = m[1]
        mz_list[i] = m[2]
    
    plt.figure()
    rx = zip(probe_dist_list, mx_list)
    ry = zip(probe_dist_list, my_list)
    rz = zip(probe_dist_list, mz_list)

    rx = sorted(rx)
    ry = sorted(ry)
    rz = sorted(rz)
    
    x_list = [x for r,x in rx]
    y_list = [y for r,y in ry]
    z_list = [z for r,z in rz]

    r_list = [r for r,x in rx]

    plt.plot(r_list, [i*1000 for i in x_list], label = 'M_x') # x1000 to get milliAmps
    plt.plot(r_list, [i*1000 for i in y_list], label = 'M_y')
    plt.plot(r_list, [i*1000 for i in z_list], label = 'M_z')
    plt.xlabel('r [m]')
    plt.ylabel('Magnetic Moment [mA*m**2]')
    plt.legend(loc='best')
    if variation_est:
        title_string = 'Using estimated dB'
    else:
        title_string = 'Using dBdI prop. const.'
    plt.title(f'Magnetic Moment - {inst} - Day {day} - {title_string}')
    plt.show()

if __name__ == "__main__":
    windows = True
    inst = 'METIS'
    var = 1
    day = 2
    variation = True

    if variation:
        if windows:
            file_path = f'.\\Results\\variation\\{inst}_var{var}_B_variation_estimated_day{day}.csv'
        else:
            file_path = os.path.expanduser(f"~/Documents/MSciProject/NewCode/Results/variation/{inst}_var{var}_B_variation_estimated.csv")

    else:
        if windows:
            file_path = f'.\\Results\\Gradient_dicts\\Day_{day}\\1hz_noorigin\\cur\\{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv'
        else:
            file_path = os.path.expanduser(f'./Results/Gradient_dicts/Day_{day}/1hz_noorigin/cur/{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv')

    solver(file_path, inst, day, variation_est = variation)

    #if use the estimated dB due to current variation, doesn't make sense as the vector dipole fit is using the actual measured B values, not just a dB.
    #hence why if using gradient dicts instead ~10x bigger magnetic moments as the estiamted dB is just dBdI prop. const. x0.1 (as METIS current var ~0.1A)

