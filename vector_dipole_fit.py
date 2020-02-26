import numpy as np
import math
import pandas as pd
from probe_dist import return_ypanel_dist, return_ypanel_loc
import os
import matplotlib.pyplot as plt
import scipy.optimize as spo

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
    mx_list_err, my_list_err, mz_list_err = 11*[0],  11*[0], 11*[0]
    for i in range(0,11):
        df_tmp = df.iloc[i]
        if variation_est:
            b = np.array([df_tmp['var_X'], df_tmp['var_Y'], df_tmp['var_Z']])
            b_err = np.array([df_tmp['var_X_err'], df_tmp['var_Y_err'], df_tmp['var_Z_err']])
        else:
            b = np.array([df_tmp['X.slope_cur'], df_tmp['Y.slope_cur'], df_tmp['Z.slope_cur']])
            b_err = np.array([df_tmp['X.slope_cur_err'], df_tmp['Y.slope_cur_err'], df_tmp['Z.slope_cur_err']])

        print('Probe ', i+1)
        loc = probe_loc_list[i]
        x, y, z = loc[0], loc[1], loc[2]
        print('x,y,z = ', x,y,z)
        r = probe_dist_list[i]
        print('r = ', r)

        xa = ((3*(x**2)/r**5)-(1/r**3))*10**(-7)
        xb = (3*y*z/r**5)*10**(-7)
        xc = (3*z*x/r**5)*10**(-7)
        xd = (3*x*y/r**5)*10**(-7)
        xe = (3*(y**2)/r**5-1/r**3)*10**(-7)
        xf = (3*z*y/r**5)*10**(-7)
        xg = (3*x*z/r**5)*10**(-7)
        xh = (3*y*z/r**5)*10**(-7)
        xi = ((3*(z**2)/r**5)-(1/r**3))*10**(-7)
        
        a = np.array([[(3*(x**2)/r**5)-(1/r**3), 3*y*z/r**5, 3*z*x/r**5],[3*x*y/r**5, 3*(y**2)/r**5-1/r**3, 3*z*y/r**5],[3*x*z/r**5, 3*y*z/r**5, (3*(z**2)/r**5)-(1/r**3)]])
        a = 10**(-7)*a
        b = 10**-9*b #get in units of Tesla
        b_err = 10**-9*b_err
        m = np.linalg.solve(a,b)


        detA = xa*((xe*xi)-(xf*xh)) - xb*((xd*xi)-(xf*xg)) + xc*((xd*xh)-(xe*xg))


        mx_err = ((1/detA)*(((((xe*xi)-(xf*xh))*b_err[0])**2)+((((xc*xh)-(xb*xi))*b_err[1])**2)+((((xb*xf)-(xc*xe))*b_err[2])**2))**0.5)*10**(7)
        my_err = ((1/detA)*(((((xf*xg)-(xd*xi))*b_err[0])**2)+((((xa*xi)-(xc*xg))*b_err[1])**2)+((((xc*xd)-(xa*xf))*b_err[2])**2))**0.5)*10**(7)
        mz_err = ((1/detA)*(((((xd*xh)-(xe*xg))*b_err[0])**2)+((((xb*xg)-(xa*xh))*b_err[1])**2)+((((xa*xe)-(xb*xd))*b_err[2])**2))**0.5)*10**(7)

        print(m) #m in units of Amp*m**2
        mx_list[i] = m[0]
        my_list[i] = m[1]
        mz_list[i] = m[2]

        mx_list_err[i] = mx_err
        my_list_err[i] = my_err
        mz_list_err[i] = mz_err

        

    plt.figure()
    rx = zip(probe_dist_list, mx_list)
    ry = zip(probe_dist_list, my_list)
    rz = zip(probe_dist_list, mz_list)

    rx = sorted(rx)
    ry = sorted(ry)
    rz = sorted(rz)

    rx_err = zip(probe_dist_list, mx_list_err)
    ry_err = zip(probe_dist_list, my_list_err)
    rz_err = zip(probe_dist_list, mz_list_err)

    rx_err = sorted(rx)
    ry_err = sorted(ry)
    rz_err = sorted(rz)
    
    x_list = [x for r,x in rx]
    y_list = [y for r,y in ry]
    z_list = [z for r,z in rz]
    x_list_err = [x for r,x in rx_err]
    y_list_err = [y for r,y in ry_err]
    z_list_err = [z for r,z in rz_err]

    r_list = [r for r,x in rx]


    def line(x,a,b):
        return a*x + b

    params_x,cov_x = spo.curve_fit(line, r_list, [i*1000 for i in x_list], sigma = [i*1000 for i in x_list_err], absolute_sigma = True)
    params_y,cov_y = spo.curve_fit(line, r_list, [i*1000 for i in y_list], sigma = [i*1000 for i in y_list_err], absolute_sigma = True)
    params_z,cov_z = spo.curve_fit(line, r_list, [i*1000 for i in z_list], sigma = [i*1000 for i in z_list_err], absolute_sigma = True)

    perr_x = np.sqrt(np.diag(cov_x))
    perr_y = np.sqrt(np.diag(cov_y))
    perr_z = np.sqrt(np.diag(cov_z))

    print(params_x[0],params_y[0],params_z[0])
    plt.scatter(r_list, [i*1000 for i in x_list], label = 'M_x')
    plt.scatter(r_list, [i*1000 for i in y_list], label = 'M_y')
    plt.scatter(r_list, [i*1000 for i in z_list], label = 'M_z')

    plt.errorbar(r_list, [i*1000 for i in x_list], yerr = [i*1000 for i in x_list_err], fmt = 'bs', markeredgewidth = 2)
    plt.errorbar(r_list, [i*1000 for i in y_list], yerr = [i*1000 for i in y_list_err], fmt = 'rs', markeredgewidth = 2)
    plt.errorbar(r_list, [i*1000 for i in z_list], yerr = [i*1000 for i in z_list_err], fmt = 'gs', markeredgewidth = 2)

    plt.plot(r_list, [params_x[0]*x + params_x[1] for x in r_list], 'b:', label = f'curve_fit - X grad: {round(params_x[0],2)} ± {round(perr_x[0],2)} int: {round(params_x[1],2)} ± {round(perr_x[1],2)}')
    plt.plot(r_list, [params_y[0]*y + params_y[1] for y in r_list], 'r:', label = f'curve_fit - Y grad: {round(params_y[0],2)} ± {round(perr_y[0],2)} int: {round(params_y[1],2)} ± {round(perr_y[1],2)}')
    plt.plot(r_list, [params_z[0]*z + params_z[1] for z in r_list], 'g:', label = f'curve_fit - Z grad: {round(params_z[0],2)} ± {round(perr_z[0],2)} int: {round(params_z[1],2)} ± {round(perr_z[1],2)}')


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
    variation = False

    if variation:
        if windows:
            file_path = f'.\\Results\\variation\\{inst}_var{var}_B_variation_estimated_day{day}.csv'
        else:
            file_path = os.path.expanduser(f"~/Documents/MSciProject/NewCode/Results/variation/{inst}_var{var}_B_variation_estimated_day{day}.csv")

    else:
        if windows:
            file_path = f'.\\Results\\Gradient_dicts\\Day_{day}\\1hz_noorigin\\cur\\{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv'
        else:
            file_path = os.path.expanduser(f'./Results/Gradient_dicts/Day_{day}/1hz_noorigin/cur/{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv')

    solver(file_path, inst, day, variation_est = variation)

    #if use the estimated dB due to current variation, doesn't make sense as the vector dipole fit is using the actual measured B values, not just a dB.
    #hence why if using gradient dicts instead ~10x bigger magnetic moments as the estiamted dB is just dBdI prop. const. x0.1 (as METIS current var ~0.1A)

