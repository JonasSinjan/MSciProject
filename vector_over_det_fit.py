import numpy as np
import math
import pandas as pd
from probe_dist import return_ypanel_dist, return_ypanel_loc, return_solohi_loc
import os
import matplotlib.pyplot as plt
import scipy.optimize as spo

def vector_linalg_lstsq(file_path, probes, sig_axes, max_current):
    df = pd.read_csv(file_path)
    df = df.iloc[:-1] #exclude probe 12

    if 'SoloHI' in file_path:
        probe_loc_list = return_solohi_loc()
    else:
        probe_loc_list = return_ypanel_loc()
    probe_loc_list = probe_loc_list[:-1] #dont want probe 12 location

    probe_dist_list, factor = return_ypanel_dist(probe_loc_list)

    #a = np.zeros((33,3))
    #b = np.zeros((33,1))
    # a = np.zeros((7, 3))
    # b = np.zeros((7, 1))
    #a = np.zeros((6,3))
    #b = np.zeros((6,1))
    #a = np.zeros((11, 3))
    #b = np.zeros((11, 1))
    #a = np.zeros((7,3))
    #b = np.zeros((7,1))
    a = np.zeros((10,3))
    b = np.zeros((10,1))
    j = 0
    for i in probes:
        #print(j)
        df_tmp = df.iloc[i]
        #b[j][0] = df_tmp['X.slope_cur']*max_current
        #b[j+1][0] = df_tmp['Y.slope_cur']*max_current
        #b[j+2][0] = df_tmp['Z.slope_cur']*max_current
       
        loc = probe_loc_list[i]
        if i == 8:
            loc = probe_loc_list[i+1]
        x, y, z = loc[0], loc[1], loc[2]
        #print('x,y,z = ', x, ',', y, ',', z)
        r = probe_dist_list[i]
        if i == 8:
            loc = probe_loc_list[i+1]
            r = probe_dist_list[i+1]
        #print('r = ', r)
        
        if 'X' in sig_axes[i]:
            b[j][0] = df_tmp['X.slope_cur']*max_current
            a[j][0] = (3*(x**2)/r**5)-(1/r**3)
            a[j][1] = 3*y*z/r**5
            a[j][2] = 3*z*x/r**5
            j += 1

        if 'Y' in sig_axes[i]:
            b[j][0] = df_tmp['Y.slope_cur']*max_current
            a[j][0] = 3*x*y/r**5
            a[j][1] = 3*(y**2)/r**5-1/r**3
            a[j][2] = 3*z*y/r**5
            j += 1

        if 'Z' in sig_axes[i]:
            b[j][0] = df_tmp['Z.slope_cur']*max_current
            a[j][0] = 3*x*z/r**5
            a[j][1] = 3*y*z/r**5
            a[j][2] = (3*(z**2)/r**5)-(1/r**3)
            j +=1 
        #if i == 9:
        #    print(a @ np.array([[14.1e-3],[-7.7e-3], [0.9e-3]]))
        #j += 3
        
    a = 10**(-7)*a
    b = 10**-9*b #get in units of Tesla
    m, rss, rank, s = np.linalg.lstsq(a,b, rcond=None)
    print(m, rss, rank, s)
    #print(rss)
    #print(rank)
    #print(s)
    mean_bx = np.mean([b for b in b[::3,0]])
    mean_by = np.mean([b for b in b[1::3,0]])
    mean_bz = np.mean([b for b in b[2::3,0]])
    mean_b = np.mean(b)
    #print(mean_bx, mean_by, mean_bz)

    # tss_x = sum([(b_i - mean_bx)**2 for b_i in b[::3,0]])
    # tss_y = sum([(b_i - mean_by)**2 for b_i in b[1::3,0]])
    # tss_z = sum([(b_i - mean_bz)**2 for b_i in b[2::3,0]]) 
    # r_2 = 1 - rss / np.sqrt(tss_x**2+tss_y**2 + tss_z**2)#np.sum((b**2))
    # N = np.identity(33) - 1/33 * np.ones((33,33))
    # test_tss = b.T @ N @ b 
    # #print('r_2 = ', r_2)
    # r2_conv = 1-rss/np.sum(b**2)
    # #print(round(r2_conv[0],3))
    # r_2_new  = 1-rss/test_tss
    # #print(r_2_new[0][0])
    # r_2_adj = 1 - ((1-r_2)*(10)/6) #3 or 1 independent variable? I think 3
    # #print(r_2_adj)
    tss = [(b_i-mean_b)**2 for b_i in b]
    r_2_test = 1 - rss/sum(tss)
    r_2_test = 1 - rss/(33*b.var())
    print('r_2 = ', r_2_test)
    return m, rss

if __name__ == "__main__":
    windows = True
    #inst = 'PHI'
    day = 1
    instru_list = ['SoloHI']#,'PHI','SWA','SoloHI','STIX','SPICE','EPD']
    if 'EUI' in instru_list:
        max_current = 0.8
    if 'METIS' in instru_list:
        max_current = 0.95
    if 'SoloHI' in instru_list and day == 2:
        max_current = 0.6
    if 'SoloHI' in instru_list and day == 1:
        max_current = 0.3

    
    #sig_probes = [0,1,2,6,7,8] #d1 metis
    #sig_probes = [0,1,2,4,5,6,7] #d2 metis

    #sig_probes = [4,5,6,8] #d1 eui
    #sig_probes = [0,3,5,6,8] #d2 eui

    sig_probes = [0,1,2,3,6,8] #d1 solohi
    #sig_probes = [1,2,3,8] #d2 solohi

    #sig_axes = {0: ['X','Z'], 1: ['X','Y'], 2: ['Y','Z'], 6: ['X','Y'], 7:['X','Y'], 8: ['X']} #d1 metis
    #sig_axes = {0: ['X','Y'], 1: ['Y'], 2: ['X'], 4: ['Y','Z'], 5: ['X','Y'], 6: ['X','Y'], 7: ['Y']} #d2 metis
    

    #sig_axes = {4: ['X'], 5: ['Y','Z'], 6: ['X','Y'], 8: ['Y']} #d1 eui
    #sig_axes = {0: ['X','Y'], 3: ['Z'], 5: ['Z'], 6: ['Y'], 8: ['Y', 'Z']} #d2 eui

    sig_axes = {0: ['Z'], 1: ['Y','Z'], 2: ['X','Y'], 3: ['X','Y'], 6: ['Y'], 8: ['X','Y']} #d1 solohi
    #sig_axes = {1: ['Z'], 2: ['Y'], 3:['X','Z'], 8 : ['Y']} #d2 solohi

    for inst in instru_list:
        #print(inst)
        if windows:
            #file_path = f'.\\Results\\Gradient_dicts\\Day_{day}\\1hz_noorigin\\cur\\{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv'
            file_path = f'.\\Results\\Gradient_dicts\\newdI_dicts\\Day_{day}\\cur\\{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv'
        else:
            #file_path = os.path.expanduser(f'./Results/Gradient_dicts/Day_{day}/1hz_noorigin/cur/{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv')
            file_path = os.path.expanduser(f'./Results/Gradient_dicts/newdI_dicts/Day_{day}/cur/{inst}_vect_dict_NOORIGIN_Day{day}_curve_fit.csv')
        #vector_linalg_lstsq(file_path, sig_probes, sig_axes, max_current)

    
    #for eui and metis - the values of x,y,z might be off due to signs, from inst to probe or
    #ibs
    x = 0.8 #0.75
    y = -0.7#-1.3
    z = -1.76
    r = 2.328

    #obs
    #x = -0.2
    #y = 0.95 #-1.05 before
    #z = -4.015
    #r = 4.155

    #for solohi
    #ibs:
    #x=-0.16
    #y=-0.7
    #z=-1.76
    #r = 1.9

    #obs
    #x=-0.75
    #y=0.95
    #z=-4.044
    #r = 4.22

    a = np.zeros((3,3))
    j=0
    a[j][0] = (3*(x**2)/r**5)-(1/r**3)
    a[j][1] = 3*y*z/r**5
    a[j][2] = 3*z*x/r**5
    a[j+1][0] = 3*x*y/r**5
    a[j+1][1] = 3*(y**2)/r**5-1/r**3
    a[j+1][2] = 3*z*y/r**5
    a[j+2][0] = 3*x*z/r**5
    a[j+2][1] = 3*y*z/r**5
    a[j+2][2] = (3*(z**2)/r**5)-(1/r**3)

    x = np.zeros((3,1))
    x[0][0] = 8.47
    x[1][0] = -6.675
    x[2][0] = -18.05

    b = 10**(-7)*a @ x/1000

    print(10**(-7)*a)
    mag = np.sqrt(b[0][0]**2 + b[1][0]**2 + b[2][0]**2)
    print(mag/10**(-9))
    

