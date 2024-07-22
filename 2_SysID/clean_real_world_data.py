import sys

sys.path.append("../gym/")
from rope import Rope

sys.path.append("../UR5e")
from CustomRobots import *

sys.path.append("../1_DataCollection")

import matplotlib.pyplot as plt

def clean_data(mocap_data, ati_data):

    UR5e = UR5eCustom()

    mocap_data_rope = mocap_data[576:1076, :, 2]
    mocap_data_base = mocap_data[500, :, 0]
    mocap_data_rope = UR5e.convert_worktraj_to_robot(mocap_data_rope, mocap_data_base)
    traj_pos = mocap_data_rope[:, [0, 2]]

    ati_data = ati_data[576:1076, 2]
    traj_force = ati_data - ati_data[0]

    return traj_pos, traj_force

def clean_raw_data():

    file_path = "../1_DataCollection/"
    folder_name = "raw_data_06122024_1400"
    count = 0

    traj_pos_save = []
    traj_force_save = []
    qf_save = []
    q0_save = []

    for file in os.listdir(file_path + folder_name):
    
        count += 1
        if not count%4 == 0: continue

        print(count, file)

        file_name = file_path + folder_name + "/" + file
        data = np.load(file_name)

        traj_pos, traj_force = clean_data(data["mocap_data_save"], data["ati_data_save"])

        traj_pos_save.append(traj_pos)
        traj_force_save.append(traj_force)
        qf_save.append(data["qf_save"])
        q0_save.append(data["q0_save"])
        
    traj_pos_save = np.array(traj_pos_save)
    traj_force_save = np.array(traj_force_save)
    qf_save = np.array(qf_save)
    q0_save = np.array(q0_save)

    plt.figure("pose X")
    plt.plot(traj_pos_save[:, :, 0].T)

    plt.figure("pose Y")
    plt.plot(traj_pos_save[:, :, 1].T)

    plt.figure("force")
    plt.plot(traj_force_save.T)
    plt.show()

    np.savez("filtered_data/test", q0_save=q0_save, qf_save=qf_save, traj_pos_save=traj_pos_save, traj_force_save=traj_force_save)

if __name__ == "__main__":
    clean_raw_data()