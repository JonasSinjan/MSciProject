import pandas as pd
import numpy as np
import scipy as sp
import glob

#set this to the directory where the data is kept on your local computer
jonas = True

if jonas:
    file_name = r'C:\Users\jonas\MSci-Data\SoloA_2019-06-21--08-10-10_20\SoloA_2019-06-19--16-19-29_1.csv'
else:
    file_name = r'your_location'#insert here

#skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
df = pd.read_csv(file_name, error_bad_lines=False, warn_bad_lines = False, skiprows = 351, sep=';')

#print(df.head())
cols = df.columns.tolist()
#print(cols)

#this will reorder the columns into the correct order
new_cols = cols[0:5] + cols[-16:-1] + [cols[-1]] + cols[13:17] + cols[9:13] + cols[5:9]
#print(new_cols)

df = df[new_cols]
#print(df.head())

#probes 10 - 10 are also missing - which means they are in the 'SoloB files'
#here I am reading the corresponding soloB file and merging it with the above dataframe
#but the start times do not match - need to calibrate - not done yet
if jonas:
    file_name = r'C:\Users\jonas\MSci-Data\SoloB_2019-06-21--08-09-10_20\SoloB_2019-06-19--16-17-38_1.csv'
else:
    file_name = r'your_location'#insert here

#skiprows is required as the read_csv function cannot read in the header of the csv files correctly for some reason - might need the header data later - need to fix
df_B = pd.read_csv(file_name, error_bad_lines=False, warn_bad_lines = False, skiprows = 170, sep=';')

#print(df_B.head())
cols = df_B.columns.tolist()

#this will reorder the columns into the correct order
new_cols = cols[9:13] + cols[1:9] + cols[13:17]
#print(new_cols)

df_B = df_B[new_cols]

df = pd.concat([df, df_B], axis = 1)

print(df.head())

#the code below was an attempt at combining all the csv files into one dataframe - but this isn't working yet so you can ignore it

# path = r'C:\Users\jonas\MSci-Data\SoloA_2019-06-21--08-10-10_20'
# all_files = glob.glob(path + "/*.csv")
#print(df.iloc[1]
#this concatenates all the csv files in one folder into one dataframe
#NB: for the unpowered test where it is moved through the array - unsure if should be concatenated
# li = []
# for filename in all_files:
#     df = pd.read_csv(filename, index_col = None, header=None, error_bad_lines=False, warn_bad_lines = False)
#     li.append(df)
# frame = pd.concat(li, ignore_index=True)
#print(frame.tail())