import pandas as pd
import csv

def correct_dI(day, inst):

    if day == 1:
        EUI_dI = [0.276, 0.096, 0.016, 0.034, 0.085, 0.082, 0.101, 0.039, -0.104, -0.104, -0.1, -0.417]
        SoloHI_dI = [0.242, 0.036, -0.014, -0.266]
        PHI_dI = [0.322, 0.056, -0.572, -0.31]
        STIX_dI = [0.108, -0.117]
        SPICE_dI = [0.484, 0.136, 0.133, -0.128, -0.146, -0.461]
        METIS_dI = [0.58, 0.129, 0.178, -0.261, -0.573]
        EPD_dI = [0.123, 0.306, 0.041, -0.352, 0.388, 0.046, 0.104, -0.592, -0.114]
        MAG_dI = [0.192, 0.018, 0.021, 0.016, -0.239]
        SWA_dI = [0.187, 0.108, 0.063, -0.065, 0.101, 0.063, -0.065, 0.134, -0.01, 0.021, -0.017, -0.121, -0.107]

    elif day == 2:
        EUI_dI = [0.291, 0.096, 0.352, -0.322, 0.091, 0.093, 0.109, 0.041, -0.109, -0.111, -0.108, -0.443]
        SoloHI_dI = [0.257, 0.301, -0.082, -0.045, -0.064, -0.112, -0.015, -0.274]
        PHI_dI = [0.326, -0.015, 0.061, 0.137, 0.047, -0.045, 0.048, -0.048, 0.152, 0.372, -0.739, -0.312]
        STIX_dI = [0.114, -0.114]
        SPICE_dI = [0.486, 0.021, 0.148, 0.138, -0.14, -0.146, -0.499]
        METIS_dI = [0.58, 0.129, 0.177, -0.3, -0.568]
        EPD_dI = [0.128, 0.324, 0.042, 0.107, -0.482, 0.325, -0.459, 0.135, 0.325, 0.043, -0.374, -0.125, 0.126, 0.325, 0.042, -0.369, 0.426, 0.046, -0.482, -0.124]
        MAG_dI = [0.192, 0.018, 0.026, 0.024, -0.255]
        SWA_dI = [0.192, 0.01, 0.019, -0.018, 0.132, -0.122, -0.206]

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
        print(inst)
        df['dI'] = new_dI
        print(df.head())

        df.to_csv(file_path)

if __name__ == "__main__":
    inst_list = ['STIX','SPICE', 'METIS', 'PHI', 'SoloHI', 'EUI', 'SWA', 'EPD']
    day = 2
    #only good for day = 2
    for inst in inst_list:
        correct_dI(day, inst)
    