import sys
import os

sys.path.append("../UR5e")
sys.path.append("../1_DataCollection")

import matplotlib.pyplot as plt

import numpy as np

from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz, sosfilt, sosfiltfilt

from CustomRobots import *

UR5e = UR5eCustom()

def butter_lowpass_filter(data, cutoff=2, fs=500.0, order=5):
    sos = butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')
    filtered = sosfiltfilt(sos, data)
    return filtered

file_path = "../1_DataCollection/"
folder_name = "raw_data"

for file in os.listdir("../1_DataCollection/" + folder_name):

    file_name = file_path + folder_name + "/" + file
    data = np.load(file_name, allow_pickle=True)

    mocap_data = data["mocap_data_save"]

    dt = 1.0/500  # 2ms
    lookahead_time = 0.05

    mocap_data_base = data["mocap_data_save"][500:1000, :, 0]

    # plt.plot(mocap_data[500:1000, :3, 1], 'b-')

    # plt.figure(3)
    # ati_data = data["ati_data_save"]

    # plt.plot(butter_lowpass_filter(ati_data.T).T[500:1000, :3], 'r-')
    # plt.plot(ati_data[500:1000, :3], 'b-')

    # continue
    # print(data["q_save"])

    q0 = data["q0_save"]
    qf = data["qf_save"]

    print(q0, qf)

    plt.figure(2)
    joint_data = data["ur5e_jointstate_data_save"][550:1050, :]
    traj = UR5e.create_trajectory(q0, qf)

    print(traj.shape)

    plt.plot(traj.T, 'r-')    
    plt.plot(joint_data, 'b-')

    plt.figure("Test 2D (Pose)")
    traj = UR5e.create_trajectory(q0, qf)
    traj = UR5e.fk_traj_stick(traj, True)
    plt.plot(traj[:, :2])

    plt.figure("Test 2D (orientation)")
    traj = UR5e.create_trajectory(q0, qf)
    traj = UR5e.fk_traj_stick(traj, True)
    plt.plot(traj[:, 2])

    # break
    # plt.plot(mocap_data[576:1076, :3, 1], 'r-')

    # traj = UR5e.create_trajectory(q0, qf)
    # traj = UR5e.fk_traj_stick_world(traj, mocap_data_base[5, :], True)


plt.show()






