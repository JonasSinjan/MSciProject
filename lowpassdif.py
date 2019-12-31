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

    print(lp_false.head())
    print(lp_true.head())

    lp_diff = lp_false.subtract(lp_true)
    lp_percent = (lp_diff/lp_false)*100

    df_percent_arr[idx] = lp_percent

tmp = 0
for df in df_percent_arr:
    for i in range(12):
        tmp += df.iloc[i]

