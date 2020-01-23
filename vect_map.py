from dB import dB
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from current import current_peaks
from probe_dist import probe_loc
import pandas as pd
import os

def vector_map(instrument, csv_file = None, vect_dict = None):
    probe_list = probe_loc()
    
    i = 0
    
    if vect_dict != None:
        soa = [0]*len(vect_dict)
        it = list(vect_dict.keys()) #list of the probe numbers starting at 1
        for i in range(len(vect_dict)):
            tmp = probe_list[int(it[i])-1]
            tmp2 = vect_dict[it[i]][:3]
            soa[i] = np.append(tmp,tmp2)
            i += 1
            
    elif csv_file != None:
        df = pd.read_csv(csv_file)
        soa = [0]*len(df)

        it = df.index
        print(it)
        for i in range(len(df)):
            tmp = probe_list[int(it[i])]
            tmp2 = df.iloc[i,[1,2,3]]
            print(tmp, tmp2)
            soa[i] = np.append(tmp,tmp2)
            i += 1

    soa = np.array(soa)
    X, Y, Z, U, V, W = zip(*soa)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    qq = ax.quiver(X, Y, Z, U, V, W)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-8, 5]) #probe 12 not shown with these limits
    ax.set_title(f'3D vector map {instrument}')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.show()



if __name__ == "__main__":
    # 

    # dict_current = current_peaks(windows, plot=False)
    # 
    # peak_datetimes = dict_current.get(f'{instrument} Current [A]')
    # current_dif = dict_current.get(f'{instrument} Current [A] dI')
    # probes = range(11)
    #vect_dict = dB(2, peak_datetimes, instrument, current_dif, windows, probes, plot=False)
    windows = True
    instrument = 'METIS'
    if windows:
        csv_filepath = f'C:\\Users\\jonas\\MSci-Code\\MSciProject\\Gradient_dicts\\Day_2\\NoOrigin\\{instrument}_vect_dict_NOORIGIN.csv'
    else:
        csv_filepath = os.path.expanduser(f'~/Documents/MSciProject/NewCode/Gradient_dicts/Day_2/NoOrigin/{instrument}_vect_dict_NOORIGIN.csv')
        
    vector_map(instrument, csv_file = csv_filepath)
    #should this read in files rather than recalculating the vect dicts?

