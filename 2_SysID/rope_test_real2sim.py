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

data = np.load("models/params_compression.npz")
print(data["x"])
params = data["x"]

rope = Rope(params, compression=True)

folder_name = "filtered_data"
file = "2.npz"
file_name = folder_name + "/" + file

data = np.load(file_name)

q0_save = data["q0_save"]
qf_save = data["qf_save"]
traj_pos_save = data["traj_pos_save"]
traj_force_save = data["traj_force_save"]

norm_mocap = 0
norm_ati = 0

cost_mocap = 0
cost_ati = 0

for i in range(q0_save.shape[0]):

    q0 = q0_save[i, :]
    qf = qf_save[i, :]
    traj_pos = traj_pos_save[i, :, :]
    traj_force = traj_force_save[i, :, :]

    success, traj_pos_sim, traj_force_sim, q_save, f_save = rope.run_sim(q0, qf)

    # render pose
    # rope.render(q_save, traj_pos_mocap, traj_pos_sim)
    # rope.render(q_save, traj_pos_mocap, traj_pos_sim, "results/" + file[:-4] + ".gif")
    
    plt.figure("pose sim v real world data")
    plt.plot(traj_pos, 'r-', label="mocap")
    plt.plot(traj_pos_sim, 'b-', label="sim")
    plt.legend()

    plt.figure("force sim v real world data")
    plt.plot(traj_force, 'k-', label="ati raw")
    plt.plot(traj_force_sim, 'b-', label="sim")
    plt.legend()

    plt.show()

    # exit()

print(norm_ati, norm_mocap)
