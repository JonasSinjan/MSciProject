import pandas as pd
import csv
import glob

def check_grads(folder_path):

    int_thresh = 0.1
    sig_thresh = 2
    #print(glob.glob(f'{folder_path}*.csv'))
    for file in glob.glob(f'{folder_path}*.csv'):
        df = pd.read_csv(file)
        #print(df.head())
        df['X_bool_sig'] = abs(df['X.slope_lin']) > sig_thresh*df['X.slope_lin_err']
        df['Y_bool_sig'] = abs(df['Y.slope_lin']) > sig_thresh*df['Y.slope_lin_err']
        df['Z_bool_sig'] = abs(df['Z.slope_lin']) > sig_thresh*df['Z.slope_lin_err']

        df['X_int_bool'] = abs(df['X_zero_err']) > int_thresh
        df['Y_int_bool'] = abs(df['Y_zero_err']) > int_thresh
        df['Z_int_bool'] = abs(df['Z_zero_err']) > int_thresh

        df = df[['Probe','X_bool_sig', 'Y_bool_sig', 'Z_bool_sig', 'X_int_bool', 'Y_int_bool', 'Z_int_bool']]
        inst = file.split('1hz_noorigin')[1]
        inst = inst[1:-28]
        #print(inst)
        df.to_csv(f'C:\\Users\\jonas\\MSci-Code\\MSciProject\\Results\\Gradient_dicts\\Day_1\\bool_check_grads\\{inst}_bool_check_day1.csv')

if __name__ == "__main__":

    fol_path = '.\\Results\\Gradient_dicts\\Day_1\\1hz_noorigin\\'
    check_grads(fol_path)
    