from read_merge import *

def powered_analysis():

    jonas = True

    if jonas:
        file_path_A = r'C:\Users\jonas\MSci-Data\powered\SoloA_2019-06-24--08-14-46_9\SoloA_2019-06-24--08-14-46_1.csv' #the first couple of files in some of the folders are from earlier days
        file_path_B = r'C:\Users\jonas\MSci-Data\powered\SoloB_2019-06-24--08-14-24_20\SoloB_2019-06-24--08-14-24_1.csv'
    else:
        file_path_A = r'your_location'#insert here
        file_path_B = r'your_location'#insert here

    df_A = soloA(file_path_A)
    df_B = soloB(file_path_B)

    df = concatenate(df_A, df_B)
    print(df.head())

    print(df[df['time']==1.00].index)

    #plotting the raw probes results
    plt.figure()
    for col in df.columns.tolist()[1:13]:
        plt.plot(df['time'], df[col], label=str(col))

    plt.xlabel('Time (s)')
    plt.ylabel('B (nT)')
    #plt.legend()
    plt.show()

powered_analysis()