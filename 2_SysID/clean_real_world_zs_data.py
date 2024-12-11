import sys

sys.path.append("../UR5e")
from CustomRobots import *

sys.path.append("../1_DataCollection")

import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, freqz, sosfilt, sosfiltfilt

def butter_lowpass_filter(data, cutoff=5, fs=500.0, order=1):
    sos = butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')
    filtered = sosfiltfilt(sos, data)
    return filtered

def clean_raw_data():

    file_path = "../1_DataCollection/"
    folder_name = "N4"

    traj_rope_tip_save = []
    traj_force_save = []
    q0_save = []
    qf_save = []

    for file in os.listdir(file_path + folder_name):

        if not file.endswith('.npz'):
            continue  # Skip files that do not end with .npz

        file_name = file_path + folder_name + "/" + file
        data = np.load(file_name)

        traj_rope_tip = data["mocap_data_robot_save"]
        # traj_rope_tip = butter_lowpass_filter(traj_rope_tip.T).T

        traj_force = data["ati_data_save"][:, 2:3]
        # traj_force = butter_lowpass_filter(traj_force.T).T

        # print(file)
        
        # plt.figure('pose')
        # plt.plot(data["mocap_data_robot_save"], label='unfiltered')
        # plt.plot(traj_rope_tip, label='filtered')
        # plt.legend()

        # plt.figure('force')
        # plt.plot(data["ati_data_save"][:, 2], label='unfiltered')
        # plt.plot(traj_force, label='filtered')
        # plt.legend()
        # plt.show()

        # plt.figure('force')
        # plt.plot(data["ati_data_save"][:, 0], label='x')
        # plt.plot(data["ati_data_save"][:, 1], label='y')
        # plt.plot(data["ati_data_save"][:, 2], label='z')
        # plt.legend()

        # plt.figure('torquee')
        # plt.plot(data["ati_data_save"][:, 3], label='x')
        # plt.plot(data["ati_data_save"][:, 4], label='y')
        # plt.plot(data["ati_data_save"][:, 5], label='z')
        # plt.legend()
        # plt.show()

        # plt.plot(data["ur5e_jointstate_data_save"], 'r.')
        # plt.plot(data["ur5e_cmd_data_save"], 'b-')
        # plt.show()

        traj_rope_tip_save.append(traj_rope_tip)
        traj_force_save.append(traj_force)
        q0_save.append(data["q0_save"])
        qf_save.append(data["qf_save"])

    traj_rope_tip_save = np.array(traj_rope_tip_save)        
    traj_force_save = np.array(traj_force_save)

    q0_save = np.array(q0_save)
    qf_save = np.array(qf_save)

    print(traj_rope_tip_save.shape)
    plt.figure("pose X")
    plt.plot(traj_rope_tip_save[:, :, 0].T)

    plt.figure("pose Y")
    plt.plot(traj_rope_tip_save[:, :, 1].T)

    print(traj_force_save.shape)
    plt.figure("force Z")
    plt.plot(traj_force_save[:, :, 0].T)

    plt.show()

    np.savez("filtered_data/" + folder_name + "", traj_rope_tip_save=traj_rope_tip_save, traj_force_save=traj_force_save, q0_save=q0_save, qf_save=qf_save)

if __name__ == "__main__":
    clean_raw_data()