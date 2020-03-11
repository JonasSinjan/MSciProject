import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ibs = pd.read_csv('Results\PowerSpectrum\Peak_files\MAG_IBS_burst_powerspectra.csv')

obs = pd.read_csv('Results\PowerSpectrum\Peak_files\MAG_OBS_burst_powerspectra.csv')

ibs = ibs[ibs['Dir'] == 'X']

obs = obs[obs['Dir'] == 'X']

#ibs_power = ibs['Ydata']
ibs['Xdata'] = round(ibs['Xdata'],1)

#obs_power = obs['Ydata']
obs['Xdata'] = round(obs['Xdata'],1)
#print(ibs.head())
ibs = ibs[ibs['Xdata'].isin(obs['Xdata'])]
#print(ibs.head())
obs = obs[obs['Xdata'] != 20.4] #for X
#print(obs.head())
#print(obs.tail())

#print(len(ibs), len(obs))

#print(ibs['Xdata'], obs['Xdata'])
print(ibs['Xdata'])
print(obs['Xdata'])
#print(ibs.loc[ibs['Xdata'].isin(obs['Xdata'])])

#print(ibs['Xdata'])
#print(obs['Xdata'])
y = np.asarray(ibs['Ydata'])/np.asarray(obs['Ydata'])
print(y)
plt.figure()
plt.plot(range(len(y)), y)
plt.show()