import pandas as pd
import csv

def correct_errs(file_path, day):

    if day == 1:
        err_path = f'.\\day1_mfsa_probe_vars.csv'
    elif day == 2:
        err_path = f'.\\day2_mfsa_probe_vars.csv'

    df = pd.read_csv(file_path)
    df_err = pd.read_csv(err_path)
    df.set_index('key', inplace=True)
    df = df[['dI', 'dB_X', 'dB_X_err', 'dB_Y', 'dB_Y_err', 'dB_Z', 'dB_Z_err' ]]
    #print(df.head())

    end_str = file_path.split('probe')[1]
    probe_num = end_str[0]
    if end_str[1] != '_':
        probe_num += end_str[1]
    print(probe_num)

    df_err = df_err.iloc[int(probe_num)-1]
    #print(df_err)

    df['dB_X_err'] = [df_err['Bx_var'] for i in range(len(df))]
    df['dB_Y_err'] = [df_err['By_var'] for i in range(len(df))]
    df['dB_Z_err'] = [df_err['Bz_var'] for i in range(len(df))]

    #print(df.head())

    
    #print(df.head())
    df.to_csv(file_path)


if __name__ == "__main__":
    #to correct the dbdI data 1hz with err files
    day = 1
    inst_list = ['METIS', 'PHI', 'SWA', 'EPD', 'SoloHI', 'STIX', 'SPICE']#'EUI'
    for inst in inst_list:
        print(inst)
        for num in range(1,13):
            file_path = f'Results\\dBdI_data\\Day{day}\\1Hz_with_err\\{inst}\\{inst}_probe{num}_vect_dict_1Hz_day{day}.csv'
            correct_errs(file_path, day)
    

