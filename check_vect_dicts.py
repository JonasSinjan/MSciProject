import pandas as pd
import glob

def check_grads(folder_path, day):

    int_thresh = 0.1
    sig_thresh = 2
    #print(glob.glob(f'{folder_path}*.csv'))
    for file in glob.glob(f'{folder_path}*.csv'):
        df = pd.read_csv(file)
        df.set_index('Probe', inplace=True)

        if 'curve' in file:
            tmp = 'cur'

            df['X_int_bool'] = abs(df['X_zero_int']) - df['X_zero_int_err'] > 0
            df['Y_int_bool'] = abs(df['Y_zero_int']) - df['Y_zero_int_err'] > 0
            df['Z_int_bool'] = abs(df['Z_zero_int']) - df['Z_zero_int_err'] > 0

        else:
            tmp = 'lin'
        
            df['X_int_bool'] = abs(df['X_zero_err']) > int_thresh
            df['Y_int_bool'] = abs(df['Y_zero_err']) > int_thresh
            df['Z_int_bool'] = abs(df['Z_zero_err']) > int_thresh

        df['X_sig_level'] = round(abs(df[f'X.slope_{tmp}'])/df[f'X.slope_{tmp}_err'],2)
        df['Y_sig_level'] = round(abs(df[f'Y.slope_{tmp}'])/df[f'Y.slope_{tmp}_err'],2)
        df['Z_sig_level'] = round(abs(df[f'Z.slope_{tmp}'])/df[f'Z.slope_{tmp}_err'],2)

        df['X_bool_sig'] = df['X_sig_level'] > sig_thresh
        df['Y_bool_sig'] = df['Y_sig_level'] > sig_thresh
        df['Z_bool_sig'] = df['Z_sig_level'] > sig_thresh

        df = df[['X_bool_sig', 'X_sig_level', 'Y_bool_sig', 'Y_sig_level', 'Z_bool_sig', 'Z_sig_level','X_int_bool', 'Y_int_bool', 'Z_int_bool']]

        df['X_sig_level'] = df['X_sig_level'].where(df['X_sig_level'] > 2, other = 0)
        df['Y_sig_level'] = df['Y_sig_level'].where(df['Y_sig_level'] > 2, other = 0)
        df['Z_sig_level'] = df['Z_sig_level'].where(df['Z_sig_level'] > 2, other = 0)
        
        inst = file.split('1hz_noorigin')[1]
        if day == 1:
            inst = inst[1:-28]
        elif day == 2:
            inst = inst[1:-23]
        print(inst)
        df.to_csv(f'C:\\Users\\jonas\\MSci-Code\\MSciProject\\Results\\Gradient_dicts\\Day_{day}\\bool_check_grads\\{inst}_bool_check_day{day}_{tmp}.csv')
        
if __name__ == "__main__":
    day = 1
    fol_path = f'.\\Results\\Gradient_dicts\\Day_{day}\\1hz_noorigin\\'
    check_grads(fol_path, day)
    