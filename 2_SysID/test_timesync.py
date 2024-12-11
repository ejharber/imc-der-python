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

    UR5e = UR5eCustom()

    for file in os.listdir(file_path + folder_name):

        if not file.endswith('.npz'):
            continue  # Skip files that do not end with .npz

        file_name = file_path + folder_name + "/" + file
        data = np.load(file_name)

        q0 = data["q0_save"]
        qf = data["qf_save"]

        traj = UR5e.create_trajectory(q0, qf, time=1).T
        plt.plot(traj, 'b.')
        # plt.plot(data["ur5e_jointstate_data_save"][100:, :], 'r.')
        plt.plot(data["ur5e_cmd_data_save"][180:, :], 'r-')
        plt.show()

        exit()

if __name__ == "__main__":
    clean_raw_data()