import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_burst_data(windows):
    if windows:
        os.environ['MFSA_raw'] = 'C:\\Users\\jonas\\MSci-Data'
    else:
        os.environ['MFSA_raw'] = os.path.expanduser('~/Documents/MsciProject/Data')
        
    project_data = os.environ.get('MFSA_raw')
    flight_data_path = os.path.join(project_data, 'BurstSpectral.mat')
    print(flight_data_path)

    mat = scipy.io.loadmat(flight_data_path)
    print(mat.keys())
    print(type(mat['ddIBS']))
    print(mat['ddIBS'].shape)
    void_arr = mat['ddOBS'][0][0] #plot obs on top of ibs to show deviation more clearly
    void_ibs = mat['ddIBS'][0][0]
    timeseries = void_arr[9]
    ibs_timeseries = void_ibs[9]
    print(ibs_timeseries.shape)

    print(timeseries.shape)
    print(len(void_arr))
    #print(mat['ddOBS'].shape)
    
    y = timeseries[:,0] #x
    y1 = timeseries[:,1] #y
    y2 = timeseries[:,2] #z 
    y3 = timeseries[:,3] #total B  OBS
    ibs_y = ibs_timeseries[:,0] #x
    ibs_y1 = ibs_timeseries[:,1] #y
    ibs_y2 = ibs_timeseries[:,2] #z 
    ibs_y3 = ibs_timeseries[:,3] #total B IBS

    #print(np.sqrt(y[0]**2 + y1[0]**2 + y2[0]**2), y3[0]) - confirms suspicion 4th column is B mag
    #print(np.sqrt(ibs_y[0]**2 + ibs_y1[0]**2 + ibs_y2[0]**2), ibs_y3[0])

    x = [x/128 for x in range(len(y))] #missing y data

    dict_d = {'Time': x, 'OBS_X': y, 'OBS_Y': y1, 'OBS_Z': y2, 'OBS_MAGNITUDE': y3, 'IBS_X': ibs_y, 'IBS_Y': ibs_y1, 'IBS_Z': ibs_y2, 'IBS_MAGNITUDE': ibs_y3 }
    df = pd.DataFrame(data=dict_d, dtype = np.float64)
    df.set_index(df['Time'], inplace=True)
    return df
    

def plot_burst(df):
    x = [x/(128*3600) for x in df.index] #128 vectors a second
    fig = plt.figure()
    plt.subplot(4,1,1)
    plt.plot(x, df['IBS_X'], label = 'IBS')
    plt.plot(x, df['OBS_X'], 'r', label = 'OBS')
    plt.legend(loc='upper right')
    plt.ylabel('Bx [nT]')
    
    plt.subplot(4,1,2)
    plt.plot(x, df['IBS_Y'], label = 'IBS')
    plt.plot(x, df['OBS_Y'], 'r',label = 'OBS')
    plt.legend(loc='upper right')
    plt.ylabel('By [nT]')
    
    plt.subplot(4,1,3)
    plt.plot(x, df['IBS_Z'], label = 'IBS')
    plt.plot(x, df['OBS_Z'], 'r', label = 'OBS')
    plt.ylabel('Bz [nT]')
    plt.legend(loc='upper right')

    plt.subplot(4,1,4)
    plt.plot(x, df['IBS_MAGNITUDE'], label = 'IBS')
    plt.plot(x, df['OBS_MAGNITUDE'], 'r', label = 'OBS')
    plt.ylabel('B [nT]')
    plt.xlabel('Time [Hours]')
    plt.legend(loc='upper right')
    
    plt.suptitle('Magnetic Field with means removed')
    plt.show()

    
def burst_powerspectra(df, OBS=True):
    if OBS:
        collist = ['Time', 'OBS_X', 'OBS_Y', 'OBS_Z']
        name_str = 'OBS_burst'
    else:
        collist = ['Time', 'IBS_X', 'IBS_Y', 'IBS_Z']
        name_str = 'IBS_burst'

    processing.powerspectrum(df, 128, collist, False, probe = 'MAG', inst = name_str)
    

def heater_data(windows):
    if windows:
        os.environ['MFSA_raw'] = 'C:\\Users\\jonas\\MSci-Data'
    else:
        os.environ['MFSA_raw'] = os.path.expanduser('~/Documents/MsciProject/Data')
        
    project_data = os.environ.get('MFSA_raw')
    flight_data_path = os.path.join(project_data, 'HeaterData.mat')
    print(flight_data_path)

    mat = scipy.io.loadmat(flight_data_path)
    print(mat.keys())

    heater = mat['ddOBS'][0][0]
    print(len(heater))

    timeseries = heater[9]
    y = timeseries[:,0]
    y1 = timeseries[:,1]
    y2 = timeseries[:,2]
    x = range(len(y))
    x = [x/16 for x in x] #16 vectors a second
    fig = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x, y)
    plt.ylabel('B [nT]')
    plt.subplot(3,1,2)
    plt.plot(x, y1)
    plt.ylabel('B [nT]')
    plt.subplot(3,1,3)
    plt.plot(x, y2)
    plt.ylabel('B [nT]')
    plt.xlabel('Time [s]')
    plt.suptitle('Magnetic Field with means removed')
    plt.show()

    heater_cur = mat['Heater'][0][0]
    print(heater_cur)

    plt.figure()
    x = [x/3600 for x in range(len(heater_cur[-1]))]
    plt.plot(x, heater_cur[-1])
    plt.ylabel('Current [A]')
    plt.xlabel('Time [Hours]')
    plt.show()

if __name__ == "__main__":
    windows = True
    burst_data(windows)