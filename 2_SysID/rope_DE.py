import sys
sys.path.append("../gym/")
from rope import Rope

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import differential_evolution

def cost_fun(params, q0_save, qf_save, traj_robot_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save, display=False):

    cost_mocap = 0
    cost_ati = 0

    for i in range(q0_save.shape[0]):

        q0 = q0_save[i, :]
        qf = qf_save[i, :]

        traj_robot = traj_robot_save[i, :, round(params[-1]):round(params[-1] + 500)]
        traj_rope_base = traj_rope_base_save[i, round(params[-2]):round(params[-2] + 500), :] # taken by mocap
        traj_rope_tip = traj_rope_tip_save[i, round(params[-2]):round(params[-2] + 500), :]
        traj_force = traj_force_save[i, round(params[-3]):round(params[-3] + 500), :]

        rope = Rope(params[:-1])
        
        success, traj_pos_sim, traj_force_sim, traj_force_sim_base, traj_force_sim_rope, q_save, _ = rope.run_sim(q0, qf)

        if not success: return 1e4
        
        cost_mocap += np.linalg.norm(traj_rope_tip - traj_pos_sim) 
        cost_ati += np.linalg.norm(traj_force - traj_force_sim)

        if display:
            print(traj_pos_sim.shape, traj_force_sim.shape, traj_force_sim_base.shape, traj_force_sim_rope.shape)

            # print(traj_force_sim_rope)
            # rope.render(q_save, traj_rope_base, traj_pos_sim)
            rope.render(q_save)

            # plt.figure("pose sim v real world data")
            # plt.plot(traj_robot, 'r-', label='real world')
            # plt.plot(traj_pos_sim, 'b-', label='sim')
            # plt.legend()

            plt.figure("force sim v real world data")
            plt.plot(traj_force, 'r.', label='real world')
            plt.plot(traj_force_sim, 'r-', label='sim total')
            plt.plot(traj_force_sim_base, 'b-', label='sim base')
            plt.plot(traj_force_sim_rope, 'g-', label='sim rope')
            plt.legend()

            plt.show()

            exit()

    cost = cost_mocap / np.linalg.norm(traj_rope_tip_save) + cost_ati / np.linalg.norm(traj_force_save)
    # cost = cost_mocap / norm_mocap
    # cost = cost_ati / norm_ati

    print(cost)

    return cost


if __name__ == "__main__":

    folder_name = "filtered_data"
    file = "N2.npz"
    file_name = folder_name + "/" + file

    data = np.load(file_name)

    traj_robot_tool_save = data["traj_robot_tool_save"]
    traj_rope_base_save = data["traj_rope_base_save"]
    traj_rope_tip_save = data["traj_rope_tip_save"]    
    traj_force_save = data["traj_force_save"]
    q0_save = data["q0_save"]
    qf_save = data["qf_save"]

    params = [2.00049247e-01, 2.00032400e-01, .1, 2, 1e8, 0.000001, 0.05, 0.05, 0.05, 60, 60, 60]
    cost_fun(params, q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save)

    bounds = [(1e-3, 0.05), (1e-3, 0.05), (1e-3, 0.05), (0.01, 100), (1e1, 1e8), (1e-5, 1), (1e-3, 1e-1), (1e-3, 1e-1), (1e-3, 1e-1), (0.1, 200), (0.1, 200), (0.1, 200)]
    
    res = differential_evolution(cost_fun, args=[q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save],                # the function to minimize
                                 bounds=bounds,
                                 maxiter=1000,
                                 workers=31,
                                 updating="deferred",
                                 init='sobol',
                                 popsize=100,
                                 tol=0.0001,
                                 disp=True)   # the random seed

    params = res.x
    params = np.power(10, params)

    np.savez("params/params_nonintertial", params=params)