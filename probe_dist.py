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


def probe_loc():
    return [probe_1, probe_2,probe_3,probe_4,probe_5,probe_6,probe_7,probe_8,probe_9,probe_10, probe_11, probe_12]


def return_ypanel_loc(): #x,y,z distance from centre of -y panel
    probe_list = probe_loc()
    y_panel = [0.045, 1, -0.326]
    
    for probe in probe_list:
        for index in [0,1,2]:
            probe[index] = probe[index] - y_panel[index]
    return probe_list

def return_solohi_loc(): #x,y,z distance from centre of -y panel
    probe_list = probe_loc()
    solohi = [1, -1, -0.326]
    
    for probe in probe_list:
        for index in [0,1,2]:
            probe[index] = probe[index] - solohi[index]
    return probe_list

def return_ypanel_dist(probe_list):
    distance = []
    factor = []
    for i in probe_list:
        tmp_dist, tmp_factor = dist(i)
        distance.append(tmp_dist)
        factor.append(tmp_factor)
    return distance, factor


if __name__ == "__main__":
    probe_list = return_ypanel_loc()
    #print(probe_list)
    distance, factor = return_ypanel_dist(probe_list)

    j = 0
    for i in range(len(probe_list)):
        tmp_dist, tmp_factor = distance[i], factor[i]
        probe_dists = probe_list[i]
        x = probe_dists[0]
        y = probe_dists[1]
        z = probe_dists[2]
        print('Probe no.', j+1, 'x = ', x, 'y = ', y, 'z = ', z, 'dist = ', round(tmp_dist,3), 'factor = ', round(tmp_factor,5))
        j += 1
 
    