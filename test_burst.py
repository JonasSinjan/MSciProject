import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ['MFSA_raw'] = 'C:\\Users\\jonas\\MSci-Data'
project_data = os.environ.get('MFSA_raw')
flight_data_path = os.path.join(project_data, 'BurstSpectral.mat')
print(flight_data_path)

mat = scipy.io.loadmat(flight_data_path, squeeze_me = True)
print(mat.keys())
#print(type(mat['ddIBS']))
#print(mat['ddIBS'].shape)

void_arr = mat['ddOBS'] #plot obs on top of ibs to show deviation more clearly
#void_ibs = mat['ddIBS'][0][0]
print(void_arr, type(void_arr))
"""
header = mat['__header__']
version = mat['__version__']
globals_mat = mat['__globals__']
function_workspace = mat['__function_workspace__'][0,:]

print(header, type(header))

print(version, type(version))


print(globals_mat, type(globals_mat))


print(function_workspace, type(function_workspace))
print(function_workspace.shape)
print(function_workspace[:15])
"""
print(void_arr[7])
print(type(void_arr[7]))
"""
timeseries = void_arr[9]
ibs_timeseries = void_ibs[9]
print(ibs_timeseries.shape)

print(timeseries.shape)
print(len(void_arr))
"""

#DATETIME NOT READABLE IN PYTHON