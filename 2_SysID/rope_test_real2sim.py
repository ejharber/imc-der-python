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

UR5e = UR5eCustom()

data = np.load("res_all_noise.npz")
print(data["x"])

params = data["x"]
# params = X = [0.05, 1, 1e-2, 1e4, 1e7, 0.15, 0.2, 0.2, 0.2]
rope = Rope(params)

file_path = "../1_DataCollection/"
folder_name = "raw_data"

from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz, sosfilt, sosfiltfilt

def butter_lowpass_filter(data, cutoff=10, fs=500.0, order=3):
    sos = butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')
    filtered = sosfiltfilt(sos, data)
    return filtered

norm_ati = 0
norm_mocap = 0
cost = 0

for file in os.listdir(file_path + folder_name):

    file_name = file_path + folder_name + "/" + file
    data = np.load(file_name)

    q0 = data["q0_save"]
    qf = data["qf_save"]
    mocap_data = data["mocap_data_save"][576:1076, :, 2]
    ati_data_raw = data["ati_data_save"]
    print(ati_data_raw.shape)
    ati_data_filtered = butter_lowpass_filter(ati_data_raw.T).T
    ati_data = ati_data_filtered[525:1025, 2]
    ati_data = ati_data - ati_data[0]
    mocap_data_base = data["mocap_data_save"][500, :, 0]
    mocap_data_rope = UR5e.convert_work_to_robot(mocap_data, mocap_data_base)
    traj_pos_mocap = mocap_data_rope[:, [0, 2]]

    traj = UR5e.create_trajectory(q0, qf)
    joint_data_fk = UR5e.fk_traj_stick(traj)

    success, traj_pos_sim, traj_force_sim, q_save, f_save = rope.run_sim(q0, qf)

    # render pose
    rope.render(q_save, traj_pos_mocap, traj_pos_sim)
    # rope.render(q_save, traj_pos_mocap, traj_pos_sim, "results/" + file[:-4] + ".gif")
    # 
    plt.figure("pose sim v real world data")
    plt.plot(traj_pos_mocap, 'r-', label="mocap")
    plt.plot(traj_pos_sim, 'b-', label="sim")
    plt.legend()

    plt.figure("force sim v real world data")
    ati_data_raw = ati_data_raw[525:1025, 2]
    plt.plot(ati_data_raw, 'k-', label="ati raw")
    # plt.plot(ati_data_filtered, 'g-', label="ati filtered")
    plt.plot(butter_lowpass_filter(ati_data), 'r-', label="ati")
    plt.plot(traj_force_sim, 'b-', label="sim")
    plt.legend()

    plt.show()

    # exit()

print(norm_ati, norm_mocap)

