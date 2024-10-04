import sys
sys.path.append("../gym/")
from rod import Rod

import matplotlib.pyplot as plt

import numpy as np

import numpy as np 
from scipy.optimize import differential_evolution

def cost_fun(params, q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_force_save):

    rod = Rod(params[:-1])

    for i in range(q0_save.shape[0]):

        q0 = q0_save[i, :]
        qf = qf_save[i, :]
        traj_robot_tool = traj_robot_tool_save[i, :, :]
        traj_rope_base = traj_rope_base_save[i, :, :]
        traj_force = traj_force_save[i, round(params[-1]):round(params[-1] + 500), :]
        
        success, traj_pos_sim, traj_force_sim_nonintertial, traj_force_sim_intertial, q_save, _ = rod.run_sim(q0, qf)

        rod.render(q_save, traj_rope_base, traj_pos_sim)

        plt.figure("pose sim v real world data")
        plt.plot(traj_rope_base, 'r-', label='real world')
        plt.plot(traj_pos_sim, 'b-', label='sim')
        plt.legend()

        plt.figure("force sim v real world data")
        plt.plot(traj_force, 'r-', label='real world')
        plt.plot(traj_force_sim_nonintertial, 'b-', label='sim noninertial')
        plt.plot(traj_force_sim_intertial, 'g-', label='sim intertial')
        plt.legend()

        plt.show()

        # exit()

if __name__ == "__main__":

    folder_name = "filtered_data"
    file = "inertial_calibration.npz"
    file_name = folder_name + "/" + file

    data = np.load(file_name)

    traj_robot_tool_save = data["traj_robot_tool_save"]
    traj_rope_base_save = data["traj_rope_base_save"]
    traj_rope_tip_save = data["traj_rope_tip_save"]    
    traj_force_save = data["traj_force_save"]
    q0_save = data["q0_save"]
    qf_save = data["qf_save"]

    params = np.load("params/inertial_calibration.npz")["params"]
    print(params)
    # exit()
    # params = [.254, .2413, 0.003, 0.03]

    cost_fun(params, q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_force_save)