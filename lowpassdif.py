import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as plt

instru_list = ['EPD', 'EUI', 'SWA', 'STIX', 'METIS', 'SPICE', 'PHI', 'SoloHI']
df_percent_arr = [0]*8

for idx, instrument in enumerate(instru_list):
    lp_false = pd.read_csv(f'./Gradient_dicts/{instrument}_vect_dict_30k.csv')
    lp_true = pd.read_csv(f'./Gradient_dicts/{instrument}_vect_dict_lowpass.csv')

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
for i in range(1,13):
    df_tmp = df_concat[df_concat.index.isin([i])]
    df_tmp['Instrument'] = instru_list
    df_tmp.loc['mean'] = df_tmp.mean()
    probe_percent_dif_lowpass_dict[f'Probe {i}'] = df_tmp
    
print(probe_percent_dif_lowpass_dict)
