import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime

filename = r'C:\Users\jonas\MSci-Data\LCL_data\Day 2 Payload LCL Current Profiles.xlsx'
df =  pd.read_excel(filename)
df.set_index(['EGSE Time'], inplace = True)
#df = df.resample(f'{1}s').mean()
#eui 9:24 - 10:10
#phi 8:03 - 8:40
#metis 10:10 - 10:57
df = df.between_time('8:03', '8:40')
print(df.head())
plt.figure()
plt.plot(df.index.time, df['PHI Current [A]'])
plt.xlabel('Time [H:M:S]')
plt.ylabel('Current [A]')
plt.title('PHI CURRENT PROFILE')
plt.show()