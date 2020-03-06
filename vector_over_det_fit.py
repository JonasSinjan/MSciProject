import numpy as np
import math
import pandas as pd
from probe_dist import return_ypanel_dist, return_ypanel_loc
import os
import matplotlib.pyplot as plt
import scipy.optimize as spo

def vector_linalg_lstsq(file_path, inst, day):
    df = pd.read_csv(file_path)
    df = df.iloc[:-1] #exclude probe 12

    probe_loc_list = return_ypanel_loc()
    probe_loc_list = probe_loc_list[:-1] #dont want probe 12 location

    probe_dist_list, factor = return_ypanel_dist(probe_loc_list)

    a = np.zeros((33,3))
    b = np.zeros((33,1))
    j = 0
    for i in range(0,11):
        print(j)
        df_tmp = df.iloc[i]
        b[j][0] = df_tmp['X.slope_cur']
        b[j+1][0] = df_tmp['Y.slope_cur']
        b[j+2][0] = df_tmp['Z.slope_cur']
       
        loc = probe_loc_list[i]
        x, y, z = loc[0], loc[1], loc[2]
        print('x,y,z = ', x, ',', y, ',', z)
        r = probe_dist_list[i]
        print('r = ', r)
        
        a[j][0] = (3*(x**2)/r**5)-(1/r**3)
        a[j][1] = 3*y*z/r**5
        a[j][2] = 3*z*x/r**5
        a[j+1][0] = 3*x*y/r**5
        a[j+1][1] = 3*(y**2)/r**5-1/r**3
        a[j+1][2] = 3*z*y/r**5
        a[j+2][0] = 3*x*z/r**5
        a[j+2][1] = 3*y*z/r**5
        a[j+2][2] = (3*(z**2)/r**5)-(1/r**3)
        
        j += 3
        
    a = 10**(-7)*a
    b = 10**-9*b #get in units of Tesla
    m, rss, rank, s = np.linalg.lstsq(a,b, rcond=None)
    print(m)
    print(rss)
    print(rank)
    print(s)
    mean_bx = np.mean([b for b in b[::3,0]])
    mean_by = np.mean([b for b in b[1::3,0]])
    mean_bz = np.mean([b for b in b[2::3,0]])
    print(mean_bx, mean_by, mean_bz)

    tss_x = sum([(b_i - mean_bx)**2 for b_i in b[::3,0]])
    tss_y = sum([(b_i - mean_by)**2 for b_i in b[1::3,0]])
    tss_z = sum([(b_i - mean_bz)**2 for b_i in b[2::3,0]]) 
    r_2 = 1 - rss / np.sqrt(tss_x**2+tss_y**2 + tss_z**2)#np.sum((b**2))
    N = np.identity(33) - 1/33 * np.ones((33,33))
    test_tss = b.T @ N @ b 
    print('r_2 = ', r_2)
    print(1-rss/np.sum((b**2)))
    print(1-rss/test_tss)
    r_2_adj = 1 - ((1-r_2)*(10)/6) #3 or 1 independent variable? I think 3
    #print(r_2_adj)
    return m, rss

if __name__ == "__main__":
    windows = True
    inst = 'PHI'
    day = 1

    if windows:
        file_path = f'.\\Results\\Gradient_dicts\\Day_{day}\\1hz_noorigin\\cur\\{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv'
    else:
        file_path = os.path.expanduser(f'./Results/Gradient_dicts/Day_{day}/1hz_noorigin/cur/{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv')

    vector_linalg_lstsq(file_path, inst, day)


