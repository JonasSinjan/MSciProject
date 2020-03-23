import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 2 Payload LCL Current Profiles.xlsx'
df =  pd.read_excel(filename)
df.set_index(['EGSE Time'], inplace = True)
inst = 'EUI'
#df = df.resample(f'{1}s').mean()
#eui 9:24 - 10:10
#phi 8:03 - 8:40
#metis 10:10 - 10:57
df = df[f'{inst} Current [A]']
df = df.between_time('9:40:30', '10:10')
print(df.head())
plt.figure()
plt.plot(df.index.time, df)
plt.xlabel('Time [H:M:S]')
plt.ylabel('Current [A]')
plt.title(f'{inst} CURRENT PROFILE')
#plt.show()

mean_val = df.mean()
print(mean_val)
df_top = df[df > mean_val]
print(df_top.head())
df_bot = df[df < mean_val]
print(df_bot.head())

top_avg = df_top.mean()
top_std = df_top.std()/np.sqrt(len(df_top))
bot_avg = df_bot.mean()
bot_std = df_bot.std()/np.sqrt(len(df_bot))

tot_std = np.sqrt(bot_std**2 + top_std**2)

dif = top_avg - bot_avg
print(dif, tot_std)

"""
# -------METIS-----------#
#metis current variation during scientific operation is 0.1286 +/- 0.0003 A

#metis only significant signal in Y at probe 10: 0.29nT/A
var = 0.29*dif
#when multiplying together, add the fractional errors in quadrature
err = np.sqrt((0.13/0.29)**2 + (tot_std/dif)**2) #fractional error in variation
print(round(var, 4), '+/-', round(err*var,4), 'nT') #in nT
#metis B var at MAG-OBS: 37 +/- 17 pT
"""

# -------EUI----------#
dif = 0.8 #rough estimate as variation not as constant as with metis
tot_std = 0.1
tot_grad = np.sqrt(0.71**2 + 1.02**2)

print(tot_grad, 'nT/A')

grad_err = np.sqrt(((0.71/tot_grad)**2)*(0.19**2) + ((1.02/tot_grad)**2)*(0.19**2))

var = dif * tot_grad

var_err = np.sqrt((grad_err/var)**2 + (tot_std/dif)**2)
print(1000*round(var, 5), '+/-', 1000*round(var_err*var,5), 'pT') #in nT