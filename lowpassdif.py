import pandas as pd
import numpy as np
import scipy as sp
from scipy.signal import butter, freqs, freqz
import matplotlib.pyplot as plt
from datetime import datetime as dt


instru_list = ['EPD', 'EUI', 'SWA', 'STIX', 'METIS', 'SPICE', 'PHI', 'SoloHI']
df_percent_arr = [0]*8

for idx, instrument in enumerate(instru_list):
    lp_false = pd.read_csv(f'./Gradient_dicts/Day_2/{instrument}_vect_dict_30k.csv')
    lp_true = pd.read_csv(f'./Gradient_dicts/Day_2/{instrument}_vect_dict_lowpass.csv')

    lp_false.set_index('Probe', inplace = True)
    lp_true.set_index('Probe', inplace = True)

    #print(lp_false.head())
    #print(lp_true.head())

    lp_diff = lp_false.subtract(lp_true)
    lp_percent = (lp_diff/lp_false)*100
    print(lp_percent.iloc[0])
    df_percent_arr[idx] = lp_percent

#make one dataframe with all the information
df_concat = pd.concat(df_percent_arr)

#filter the overall df for the desired probe

probe_percent_dif_lowpass_dict = {}
time_test = dt.now()

for i in range(1,13):
    df_tmp = df_concat[df_concat.index.isin([i])]
    df_tmp['Instrument'] = instru_list
    df_tmp.loc['std'] = df_tmp[df_tmp.index.isin([i])].std()/np.sqrt(8)
    df_tmp.loc['mean'] = df_tmp[df_tmp.index.isin([i])].mean()
    probe_percent_dif_lowpass_dict[f'Probe {i}'] = df_tmp
    
    
#print(probe_percent_dif_lowpass_dict)
time_test = dt.now() - time_test
print(time_test)

# df = probe_percent_dif_lowpass_dict.get('Probe 1')
# x = df.columns.tolist()[:-1]
# y = df.iloc[-1][:-1]

#fig, ax = plt.subplots()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(),2)
        #print(height)
        if height < 0 :
            xytext1 = (0, -14)
        else:
            xytext1 = (0, 3)
        plt.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 4, height),
                    xytext=xytext1,  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
def butter_lowpass(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
        
b, a = butter_lowpass(15, 1000) 
w, h = freqs(b, a)
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)))
#plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
#plt.margins(0, 0.1)
plt.show()

for i in range(12): 
 
    df = probe_percent_dif_lowpass_dict.get(f'Probe {i+1}')
    x = df.columns.tolist()[:-1]
    y = df.iloc[-1][:-1]
    error = df.iloc[-2][:-1]
    plt.figure(figsize = (15.0, 8.0))
    rects1 = plt.bar(x, y, yerr = error, capsize = 10)
    plt.ylabel('Percent difference with low pass filter (no filter is reference)')
    plt.title(f'Probe {df.index[0]}')

    autolabel(rects1)
    plt.tight_layout()
    plt.show()
