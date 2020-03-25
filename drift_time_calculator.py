from datetime import datetime, timedelta
import numpy as np

def hours_minutes_seconds_mircoseconds(td):
    return td.seconds//3600, (td.seconds//60)%60, td.seconds - (td.seconds//3600)*3600 - ((td.seconds//60)%60)*60, td.microseconds

#Day_1
print('Day 1')

#mfsa
#print('MFSA')
mfsa_start_one = datetime(2019,6,21,10,58,18,293000)
mfsa_start_two = datetime(2019,6,21,10,58,28,521000)
mfsa_start_three = datetime(2019,6,21,10,58,38,378000)

mfsa_end_one = datetime(2019,6,21,16,3,18,235000)
mfsa_end_two = datetime(2019,6,21,16,3,28,660000)
mfsa_end_three = datetime(2019,6,21,16,3,38,528000)

mfsa_dif_one = mfsa_end_one - mfsa_start_one
#print(hours_minutes_seconds_mircoseconds(mfsa_dif_one))

mfsa_dif_two = mfsa_end_two - mfsa_start_two
#print(hours_minutes_seconds_mircoseconds(mfsa_dif_two))

mfsa_dif_three = mfsa_end_three - mfsa_start_three
#print(hours_minutes_seconds_mircoseconds(mfsa_dif_three))

#mag
#print('MAG')
mag_start_one = datetime(2019,6,21,8,58,48,254300)
mag_start_two = datetime(2019,6,21,8,58,58,488700)
mag_start_three = datetime(2019,6,21,8,59,8,348000)

mag_end_one = datetime(2019,6,21,14,3,47,992000)
mag_end_two = datetime(2019,6,21,14,3,58,422000)
mag_end_three = datetime(2019,6,21,14,4,8,289000)


mag_dif_one = mag_end_one - mag_start_one
#print(hours_minutes_seconds_mircoseconds(mag_dif_one))

mag_dif_two = mag_end_two - mag_start_two
#print(hours_minutes_seconds_mircoseconds(mag_dif_two))

mag_dif_three = mag_end_three - mag_start_three
#print(hours_minutes_seconds_mircoseconds(mag_dif_three))

print('Diff')
print(mfsa_dif_one-mag_dif_one)
print(mfsa_dif_two-mag_dif_two)
print(mfsa_dif_three-mag_dif_three)

def calc_mean_std(L):
    n = len(L)
    mean = sum(L) / float(n)
    dev = [(x - mean) for x in L]
    print(dev)
    dev2 = [x*x for x in dev]
    return mean, np.sqrt(sum(dev2) / n) 

timedeltas = [(mfsa_dif_one-mag_dif_one).total_seconds(), (mfsa_dif_two-mag_dif_two).total_seconds(), (mfsa_dif_three-mag_dif_three).total_seconds()]
avg_td, std_td = calc_mean_std(timedeltas)
print('Mean', 'Std')
print(avg_td, std_td)

#Day_2
print('Day 2')

#mfsa
#print('MFSA')
mfsa_start_one = datetime(2019,6,24,10,58,18,293000)
mfsa_start_two = datetime(2019,6,24,10,58,28,521000)
mfsa_start_three = datetime(2019,6,24,10,58,38,378000)

mfsa_end_one = datetime(2019,6,24,16,3,18,235000)
mfsa_end_two = datetime(2019,6,24,16,3,28,660000)
mfsa_end_three = datetime(2019,6,24,16,3,38,528000)

mfsa_dif_one = mfsa_end_one - mfsa_start_one
#print(hours_minutes_seconds_mircoseconds(mfsa_dif_one))

mfsa_dif_two = mfsa_end_two - mfsa_start_two
#print(hours_minutes_seconds_mircoseconds(mfsa_dif_two))

mfsa_dif_three = mfsa_end_three - mfsa_start_three
#print(hours_minutes_seconds_mircoseconds(mfsa_dif_three))

#mag
#print('MAG')
mag_start_one = datetime(2019,6,24,8,58,48,254300)
mag_start_two = datetime(2019,6,24,8,58,58,488700)
mag_start_three = datetime(2019,6,24,8,59,8,348000)

mag_end_one = datetime(2019,6,24,14,3,47,992000)
mag_end_two = datetime(2019,6,24,14,3,58,422000)
mag_end_three = datetime(2019,6,24,14,4,8,289000)


mag_dif_one = mag_end_one - mag_start_one
#print(hours_minutes_seconds_mircoseconds(mag_dif_one))

mag_dif_two = mag_end_two - mag_start_two
#print(hours_minutes_seconds_mircoseconds(mag_dif_two))

mag_dif_three = mag_end_three - mag_start_three
#print(hours_minutes_seconds_mircoseconds(mag_dif_three))

print('Diff')
print(mfsa_dif_one-mag_dif_one)
print(mfsa_dif_two-mag_dif_two)
print(mfsa_dif_three-mag_dif_three)

timedeltas = [(mfsa_dif_one-mag_dif_one).total_seconds(), (mfsa_dif_two-mag_dif_two).total_seconds(), (mfsa_dif_three-mag_dif_three).total_seconds()]
avg_td, std_td = calc_mean_std(timedeltas)
print('Mean', 'Std')
print(avg_td, std_td)