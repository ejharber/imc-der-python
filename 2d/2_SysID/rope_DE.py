import sys
sys.path.append("../gym/")
from rope import Rope

sys.path.append("../UR5e")
sys.path.append("../1_DataCollection")

import matplotlib.pyplot as plt

import numpy as np

from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz, sosfilt, sosfiltfilt

from CustomRobots import *

import numpy as np 
from scipy.optimize import differential_evolution


def butter_lowpass_filter(data, cutoff=5, fs=500.0, order=5):
    sos = butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')
    filtered = sosfiltfilt(sos, data)
    return filtered

def cost_fun(params):

    UR5e = UR5eCustom()
    rope = Rope(params)

    file_path = "../1_DataCollection/"
    folder_name = "raw_data"

    norm_mocap = 0
    norm_ati = 0

    cost_mocap = 0
    cost_ati = 0

    count = 0
    for file in os.listdir(file_path + folder_name):
        count += 1

        if not count%4 == 0: continue

        file_name = file_path + folder_name + "/" + file
        data = np.load(file_name)

        q0 = data["q0_save"]
        qf = data["qf_save"]
        mocap_data = data["mocap_data_save"][576:1076, :, 2]
        ati_data = data["ati_data_save"]
        ati_data = butter_lowpass_filter(ati_data.T).T[525:1025, :3]
        ati_data = np.linalg.norm(ati_data, axis=1)
        ati_data = ati_data - ati_data[0]
        mocap_data_base = data["mocap_data_save"][500, :, 0]
        mocap_data_rope = UR5e.convert_work_to_robot(mocap_data, mocap_data_base)
        traj_pos_mocap = mocap_data_rope[:, [0, 2]]

        traj = UR5e.create_trajectory(q0, qf)
        joint_data_fk = UR5e.fk_traj_stick(traj)

        success, traj_pos_sim, traj_force_sim, _, _ = rope.run_sim(q0, qf)

        norm_mocap += np.linalg.norm(traj_pos_mocap) 
        norm_ati += np.linalg.norm(ati_data)
        
        if not success:
            return 1e8
            # cost_mocap = 1e8
            break

        cost_mocap += np.linalg.norm(traj_pos_mocap - traj_pos_sim)        
        cost_ati += np.linalg.norm(traj_force_sim - ati_data)        

        # mocap_data = data["mocap_data_save"][576:1076, :, 2]

    #     plt.figure("pose sim v real world data")
    #     plt.plot(mocap_data_rope[:, :2], 'r-')
    #     plt.plot(traj_pos_sim, 'b-')

    #     plt.figure("force sim v real world data")
    #     plt.plot(ati_data[:], 'r-')
    #     plt.plot(traj_force_sim, 'b-')

    # plt.show()

    cost = cost_mocap / norm_mocap # + cost_ati / norm_ati

    return cost

params = [0.05, 1e-2, 1e7, 0.15, 0.2, 0.2, 0.2]
print(cost_fun(params))

res = differential_evolution(cost_fun,                  # the function to minimize
                             [(5e-3, 1e-1), (1e-8, 1), (1e5, 1e9), (1e-3, 1), (1e-3, 2), (1e-3, 2), (1e-3, 2)],
                             maxiter=20,
                             workers=2,
                             disp=True)   # the random seed

print(res)

np.savez("res_noforce", x=res.x)
