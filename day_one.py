import matplotlib.pyplot as plt
from read_merge import soloA, soloB, read_files
import pandas as pd

def day_one():
    #set this to the directory where the data is kept on your local computer
    jonas = True

    if jonas:
        file_path_A = r'C:\Users\jonas\MSci-Data\day_one\SoloA_2019-06-21--08-10-10_20\SoloA_2019-06-21--08-10-10_1.csv'
        file_path_B = r'C:\Users\jonas\MSci-Data\day_one\SoloB_2019-06-21--08-09-10_20\SoloB_2019-06-21--08-09-10_1.csv'
        path_A = r'C:\Users\jonas\MSci-Data\day_one\SoloA_2019-06-21--08-10-10_20'
        path_B = r'C:\Users\jonas\MSci-Data\day_one\SoloB_2019-06-21--08-09-10_20'
    else:
        file_path_A = r'your_location'#insert here
        file_path_B = r'your_location'#insert here

    soloA = True
    df = read_files(path_A, soloA)
    print(df.tail())
    print(len(df))
    #yolo
    
    # df_A = soloA(file_path_A)
    # df_B = soloB(file_path_B)

    # df = concatenate(df_A, df_B)
    # print(df.head())

    #print(df[df['time']==1.00].index) #returns index 20 - proves that this data file is already sampled at 20Hz.

    plot = False

    if plot:
        #plotting the raw probes results
        plt.figure()
        for col in df.columns.tolist()[1:5]:
            plt.plot(df['time'], df[col], label=str(col))
            
        # for col in df.columns.tolist()[-4:0]:
        #     plt.plot(df['time'], df[col], label=str(col))

        plt.xlabel('Time (s)')
        plt.ylabel('B (nT)')
        #plt.legend()
        plt.show()

    
day_one()

