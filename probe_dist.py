probe_1 = [0.045,0,2.674]
probe_2 = [0.045,-2.12,1.794]
probe_3 = [0.045,-2.12,-0.326]
probe_4 = [0.045,-2.12,-1.846]
probe_5 = [0.045,0,-3.326]
probe_6 = [0.045,2.12,-1.846]
probe_7 = [0.045,2.12,-0.326]
probe_8 = [0.045,2.12,1.794]
probe_9 = [0.84,-0.3,-2.086]
probe_10 = [0.245,-0.05,-4.341]
probe_11 = [0.685,-0.175,-4.211]
probe_12 = [7.664,11.092,-1.283]

def dist(probe):
    #returns dist and 1/r^3 for dipole power law
    tmp = 0
    for i in probe:
        tmp += i**2
    return tmp**(1/2), 1/(tmp**(3/2))

probe_list = [probe_1, probe_2,probe_3,probe_4,probe_5,probe_6,probe_7,probe_8,probe_9,probe_10, probe_11, probe_12]
j = 0
for i in probe_list:
    tmp_dist, tmp_factor = dist(i)
    print('Probe no.', j+1, 'dist = ', round(tmp_dist,3), 'factor = ', round(tmp_factor,5))
    j += 1