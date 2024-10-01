import sys

sys.path.append("../UR5e")
from CustomRobots import *

sys.path.append("../1_DataCollection")

import matplotlib.pyplot as plt

def clean_data(robot_data, mocap_data, ati_data, q0, qf):

    UR5e = UR5eCustom()
    traj = UR5e.create_trajectory(q0, qf, time=1)
    traj = UR5e.fk_traj_stick(traj, True)

    # print(robot_data.shape)
    traj_robot_tool = robot_data[500:1000, [0 ,2]]

    mocap_data_rope_base = mocap_data[500:1200, :, 1]
    mocap_data_base = mocap_data[500, :, 0]
    mocap_data_rope_base = UR5e.convert_worktraj_to_robot(mocap_data_rope_base, mocap_data_base)
    traj_rope_base = mocap_data_rope_base[:, [0, 2]]

    # plt.figure("pose traj")
    # plt.plot(traj[:, [0, 1]], 'r-', label='sim')
    # plt.plot(traj_rope_base, 'b-', label='mocap')
    # plt.legend()
    # # plt.show()
    # exit()

    mocap_data_rope = mocap_data[500:1200, :, 2]
    mocap_data_base = mocap_data[500, :, 0]
    mocap_data_rope = UR5e.convert_worktraj_to_robot(mocap_data_rope, mocap_data_base)
    traj_rope_tip = mocap_data_rope[:, [0, 2]]
    # traj_rope_tip = np.copy(traj_rope_base) * 0

    # plt.figure("force traj")
    # plt.plot(ati_data[:, 3:])
    # plt.show()

    traj_force = ati_data[500:1200, 2]
    # traj_force = ati_data - ati_data[0]

    # plt.figure("force traj")
    # plt.plot(traj_force)

    # plt.show()

    # plt.plot(traj_robot)
    # plt.plot(traj_rope_base)
    # # plt.plot(ati_data)
    # plt.show()

    return traj_robot_tool, traj_rope_base, traj_rope_tip, traj_force

def clean_raw_data():

    file_path = "../1_DataCollection/"
    folder_name = "raw_data_inertial_calibration"

    traj_robot_tool_save = []
    traj_rope_base_save = []
    traj_rope_tip_save = []
    traj_force_save = []
    q0_save = []
    qf_save = []

    for file in os.listdir(file_path + folder_name):

        print(file)

        file_name = file_path + folder_name + "/" + file
        data = np.load(file_name)

        traj_robot_tool, traj_rope_base, traj_rope_tip, traj_force = clean_data(data["ur5e_tool_data_save"], data["mocap_data_save"], data["ati_data_save"], data["q0_save"], data["qf_save"])

        # if np.any(traj_force > 100) or np.any(traj_force < -100): continue

        traj_robot_tool_save.append(traj_robot_tool)
        traj_rope_base_save.append(traj_rope_base)
        traj_rope_tip_save.append(traj_rope_tip)
        traj_force_save.append(traj_force)
        q0_save.append(data["q0_save"])
        qf_save.append(data["qf_save"])

    traj_robot_tool_save = np.array(traj_robot_tool_save)
    traj_rope_base_save = np.array(traj_rope_base_save)
    traj_rope_tip_save = np.array(traj_rope_tip_save)        
    traj_force_save = np.expand_dims(traj_force_save, axis=2)
    q0_save = np.array(q0_save)
    qf_save = np.array(qf_save)

    plt.figure("pose X")
    plt.plot(traj_robot_tool_save[:, :, 0].T)
    plt.plot(traj_rope_base_save[:, :, 0].T)

    plt.figure("pose Y")
    plt.plot(traj_robot_tool_save[:, :, 0].T)
    plt.plot(traj_rope_base_save[:, :, 0].T)

    # plt.figure("force")
    # plt.plot(traj_force_save[:, :, 0].T)
    # plt.show()

    print(traj_robot_tool_save.shape, traj_rope_base_save.shape, traj_rope_tip_save.shape, traj_force_save.shape, q0_save.shape, qf_save.shape)

    np.savez("filtered_data/inertial_calibration.npz", traj_robot_tool_save=traj_robot_tool_save, traj_rope_base_save=traj_rope_base_save, traj_rope_tip_save=traj_rope_tip_save, 
                                                  traj_force_save=traj_force_save, q0_save=q0_save, qf_save=qf_save)

if __name__ == "__main__":
    clean_raw_data()