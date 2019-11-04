import matplotlib.pyplot as plt
from read_merge import soloA, soloB, read_files, powerspecplot
from align import align
import pandas as pd
import os

def day_two(collist, soloA_bool):

    jonas = True

    if jonas:
        file_path_A = r'C:\Users\jonas\MSci-Data\powered\SoloA_2019-06-24--08-14-46_9\SoloA_2019-06-24--08-14-46_1.csv' #the first couple of files in some of the folders are from earlier days
        file_path_B = r'C:\Users\jonas\MSci-Data\powered\SoloB_2019-06-24--08-14-24_20\SoloB_2019-06-24--08-14-24_1.csv'
        path_A = r'C:\Users\jonas\MSci-Data\day_two\SoloA_2019-06-24--08-14-46_9'
        path_B = r'C:\Users\jonas\MSci-Data\day_two\SoloB_2019-06-24--08-14-24_20'
    else:
        file_path_A = os.path.expanduser("~/Documents/MsciProject/Data/SoloA_2019-06-24--08-14-46_9/SoloA_2019-06-24--08-14-46_1.csv")
        file_path_B = os.path.expanduser("~/Documents/MsciProject/Data/SoloB_2019-06-24--08-14-24_20/SoloB_2019-06-24--08-14-24_1.csv")

    # df_A = soloA(file_path_A)
    # df_B = soloB(file_path_B)

    if soloA_bool:
        df = read_files(path_A, soloA_bool, jonas, collist)
    else:
        df = read_files(path_B, soloA_bool, jonas, collist)
    print(len(df))
    
    time_diff = align(file_path_A, file_path_B)
    print(time_diff)

    # df = concatenate(df_A, df_B)
    # print(df.head())

    #print(df[df['time']==1.00].index)

    # #plotting the raw probes results
    # plt.figure()
    # for col in df.columns.tolist()[1:13]:
    #     plt.plot(df['time'], df[col], label=str(col))

    # plt.xlabel('Time (s)')
    # plt.ylabel('B (nT)')
    # plt.legend()
    # plt.show()

    fs = 500 # sampling rate
    powerspecplot(df, fs, collist)

if __name__ == "__main__":
    num = '02'
    collist = ['time', f'Probe{num}_X', f'Probe{num}_Y', f'Probe{num}_Z', f'Probe{num}_||']
    soloA_bool = True
    day_two(collist, soloA_bool)