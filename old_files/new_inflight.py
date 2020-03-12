import scipy.io
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

start = time.time()
os.environ['MFSA_raw'] = 'C:\\Users\\jonas\\MSci-Data'

project_data = os.environ.get('MFSA_raw')
flight_data_path = os.path.join(project_data, 'BurstSpectral2.mat')
#print(flight_data_path)

mat = scipy.io.loadmat(flight_data_path)
#print(mat.keys())
#print(type(mat['ddIBS']))
#print(mat['ddIBS'].shape)
void_arr = mat['ddOBS'][0][0] #plot obs on top of ibs to show deviation more clearly
void_ibs = mat['ddIBS'][0][0]
print(void_arr)
print(void_ibs)
timeseries = void_arr[9]
ibs_timeseries = void_ibs[9]

end = 128*3600*24
y = timeseries[:end,0] #x
y1 = timeseries[:end,1] #y
y2 = timeseries[:end,2] #z 
y3 = timeseries[:end,3] #total B  OBS
ibs_y = ibs_timeseries[:end,0] #x
ibs_y1 = ibs_timeseries[:end,1] #y
ibs_y2 = ibs_timeseries[:end,2] #z 
ibs_y3 = ibs_timeseries[:end,3] #total B IBS

x = [round(x/128,3) for x in range(len(y))]
dict_d = {'OBS_X': y, 'OBS_Y': y1, 'OBS_Z': y2, 'OBS_MAGNITUDE': y3, 'IBS_X': ibs_y, 'IBS_Y': ibs_y1, 'IBS_Z': ibs_y2, 'IBS_MAGNITUDE': ibs_y3 }
df = pd.DataFrame(data=dict_d, dtype = np.float64)
#end = datetime(2020,3,3,15,58,46) + timedelta(seconds = 42463, microseconds=734375)
    
#date_range = pd.date_range(start = datetime(2020,3,3,15,58,46,0), end = end, freq='7812500ns') #1/128 seconds exactly for 1/16 just need microseconds 'ms'
#df.set_index(date_range[:-1], inplace=True) #for some reason, one extra time created
#print(df.head())

print('df successfully loaded\nExecution time: ', round(time.time() - start,3), ' seconds')

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