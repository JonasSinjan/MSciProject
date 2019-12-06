from dB import dB
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from current import current_peaks
from probe_dist import probe_loc

def vector_map(vect_dict, instrument):
    probe_list = probe_loc()
    soa = [0]*len(vect_dict)
    i = 0

    it = list(vect_dict.keys())
    for i in range(len(vect_dict)):
        tmp = probe_list[int(it[i])-1]
        tmp2 = vect_dict[it[i]]
        soa[i] = np.append(tmp,tmp2)
        i += 1

    soa = np.array(soa)
    X, Y, Z, U, V, W = zip(*soa)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    qq = ax.quiver(X, Y, Z, U, V, W)
    ax.set_xlim([-4, 10])
    ax.set_ylim([-5, 15])
    ax.set_zlim([-8, 8])
    ax.set_title(f'3D vector map {instrument}')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.show()



if __name__ == "__main__":
    jonas = True

    dict_current = current_peaks(jonas, plot=False)
    instrument = 'EUI'
    peak_datetimes = dict_current.get(f'{instrument} Current [A]')
    current_dif = dict_current.get(f'{instrument} Current [A] dI')
    probes = range(12)
    vect_dict = dB(peak_datetimes, instrument, current_dif, jonas, probes, plot=False)
    vector_map(vect_dict, instrument)

