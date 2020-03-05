import scipy.io
import os
import matplotlib.pyplot as plt

def burst_data(windows):
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

    print(timeseries.shape)
    print(len(void_arr))
    #print(mat['ddOBS'].shape)

    y = timeseries[:,0] #x
    y1 = timeseries[:,1] #y
    y2 = timeseries[:,2] #z 
    y3 = timeseries[:,3] #current?
    ibs_y = ibs_timeseries[:,0] #x
    ibs_y1 = ibs_timeseries[:,1] #y
    ibs_y2 = ibs_timeseries[:,2] #z 

    x = range(len(y))
    x = [x/(128*3600) for x in x] #128 vectors a second
    fig = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x, ibs_y, label = 'IBS')
    plt.plot(x, y, 'r', label = 'OBS')
    plt.legend(loc='upper right')
    plt.ylabel('Bx [nT]')
    
    plt.subplot(3,1,2)
    plt.plot(x, ibs_y1, label = 'IBS')
    plt.plot(x, y1, 'r',label = 'OBS')
    plt.legend(loc='upper right')
    plt.ylabel('By [nT]')
    
    plt.subplot(3,1,3)
    plt.plot(x, ibs_y2, label = 'IBS')
    plt.plot(x, y2, 'r', label = 'OBS')
    plt.ylabel('Bz [nT]')
    plt.xlabel('Time [Hours]')
    plt.legend(loc='upper right')
    
    plt.suptitle('Magnetic Field with means removed')
    #fig.tight_layout()
    plt.figure()
    plt.plot(x, y3)
    plt.show()


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
    plt.plot(range(len(heater_cur[-1])), heater_cur[-1])
    plt.show()

if __name__ == "__main__":
    windows = True
    burst_data(windows)