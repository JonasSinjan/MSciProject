import pandas as pd
import csv

def correct_dI(day, inst):

    EUI_dI = [0.28, 0.096, 0.345, -0.3, 0.084, 0.083, 0.103, 0.042, -0.1, -0.103, -0.101, -0.428]
    SoloHI_dI = [0.257, 0.299, -0.082, -0.045, -0.062, -0.105, -0.015, -0.257]
    PHI_dI = [0.309, -0.018, 0.055, 0.128, 0.05, -0.044, 0.047, -0.047, 0.146, 0.332, -0.697, -0.291]
    STIX_dI = [0.106, -0.113]
    SPICE_dI = [0.484, 0.025, 0.139, 0.132, -0.137, -0.148, -0.499]
    METIS_dI = [0.582, 0.129, 0.177, -0.283, -0.569]
    EPD_dI = [0.129, 0.293, 0.044, 0.1, -0.479, 0.32, -0.463, 0.122, 0.314, 0.039, -0.37, -0.119, 0.127, 0.319, 0.04, -0.372, 0.409, 0.043, -0.481, -0.125]
    MAG_dI = [0.191, 0.019, 0.026, 0.022, -0.242]
    SWA_dI = [0.186, 0.01, 0.017, -0.017, 0.132, -0.113, -0.206]

    for num in range(1,13):
        file_path = f'Results\\dBdI_data\\new_dI_copy\\Day{day}\\{inst}\\{inst}_probe{num}_vect_dict_1Hz_day{day}.csv'

        df = pd.read_csv(file_path)
        df.set_index('key', inplace=True)
        if inst == 'EUI':
            new_dI = EUI_dI
        elif inst == 'SoloHI':
            new_dI = SoloHI_dI
        elif inst == 'PHI':
            new_dI = PHI_dI
        elif inst == 'STIX':
            new_dI = STIX_dI
        elif inst == 'SPICE':
            new_dI = SPICE_dI
        elif inst == 'METIS':
            new_dI = METIS_dI
        elif inst == 'EPD':
            new_dI = EPD_dI
        elif inst == 'MAG':
            new_dI = MAG_dI
        elif inst == 'SWA':
            new_dI = SWA_dI

        df['dI'] = new_dI
        print(df.head())

        df.to_csv(file_path)

if __name__ == "__main__":
    inst_list = ['STIX','SPICE', 'PHI', 'SoloHI', 'EUI', 'SWA', 'EPD']
    day = 2

    for inst in inst_list:
        correct_dI(day, inst)
    