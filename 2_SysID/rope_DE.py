import sys
sys.path.append("../gym/")
from rope import Rope

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import differential_evolution

def cost_fun(params, inertial_params, q0_save, qf_save, traj_robot_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save, display=False):

    cost_mocap_x = 0
    cost_mocap_y = 0
    cost_ati = 0

    if params[1] > params[0]:
        print()
        return 1e4

    # params = [inertial_params[0], inertial_params[1], .1, 2, 1e8, 0.000001, inertial_params[3], 0.05, 0.05, inertial_params[4], 60, 60]
    # params = [inertial_params[0], inertial_params[1], params[0], params[1], params[2], params[3], 
    #                     inertial_params[3], params[4], params[5], inertial_params[4], params[6]]

    for i in range(q0_save.shape[0]):

        q0 = q0_save[i, :]
        qf = qf_save[i, :]

        traj_robot = traj_robot_save[i, :, :]
        traj_rope_base = traj_rope_base_save[i, round(params[-1]):round(params[-1] + 500), :] # taken by mocap
        traj_rope_tip = traj_rope_tip_save[i, round(params[-1]):round(params[-1] + 500), :]
        traj_force = traj_force_save[i, round(params[-2]):round(params[-2] + 500), :]

        rope = Rope(params[:-1])
        
        success, traj_pos_sim, traj_force_sim, traj_force_sim_base, traj_force_sim_rope, q_save, _ = rope.run_sim(q0, qf)

        if not success: 
            print()
            return 1e4

        # print(traj_pos_sim.shape, traj_force_sim.shape, traj_force_sim_base.shape, traj_force_sim_rope.shape, traj_force.shape, traj_rope_tip.shape)
        
        # print(traj_rope_tip.shape, traj_pos_sim)
        cost_mocap_x += np.linalg.norm(traj_rope_tip[:, 0] - traj_pos_sim[:, 0]) 
        cost_mocap_y += np.linalg.norm(traj_rope_tip[:, 1] - traj_pos_sim[:, 1]) 
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

    # print(traj_rope_tip_save.shape)
    cost = (cost_mocap_x / (np.max(traj_rope_tip_save[:, :, 0]) - np.min(traj_rope_tip_save[:, :, 0])) +  
           cost_mocap_y / (np.max(traj_rope_tip_save[:, :, 1]) - np.min(traj_rope_tip_save[:, :, 1])) + 
           cost_ati / (np.max(traj_force_save) - np.min(traj_force_save)))
    # cost = cost_mocap / norm_mocap
    # cost = cost_ati / norm_ati

    print(cost)
    print(params)
    print()

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

    inertial_params = np.load("params/inertial_calibration.npz")["params"]
    print(inertial_params)
    # exit()
    # params = [.254, .2413, 0.003, 0.03]

    # params = [inertial_params[0], inertial_params[1], .1, 2, 1e8, 0.000001, inertial_params[3], 0.05, 0.05, inertial_params[4], 60, 60]
    # bounds = [(0.5842 - 0.1, 0.5842 + 0.1), 
    #           (0.001, 1), 
    #           (1, 1e5), 
    #           (1e-5, 1), 
    #           (.0103 - 50/1000, .0103 + 50/1000), 
    #           (.06880 - 50/1000, .06880 + 50/1000), 
    #           (30, 70)]


    bounds = [(0.01, 0.4), (0.01, 0.4), (0.5842 - 0.1, 0.5842 + 0.1), # DLs
              (0.001, 1), # Kb
              (0.001, 1e2),
              (1, 1e5), # Ks
              (1, 1e5), # Ks
              (1e-5, 1), # damp
              (0.028, 0.1), (.0103 - 50/1000, .0103 + 50/1000), (.06880 - 50/1000, .06880 + 50/1000), # mass
              (0.1, 200), (0.1, 200)] # time sync


    # bounds = [(1e-3, 0.05), (1e-3, 0.05), (1e-3, 0.05), (0.01, 100), (1e1, 1e8), (1e-5, 1), (1e-3, 1e-1), (1e-3, 1e-1), (1e-3, 1e-1), (0.1, 200), (0.1, 200), (0.1, 200)]
    
    res = differential_evolution(cost_fun, args=[inertial_params, q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save],                # the function to minimize
                                 bounds=bounds,
                                 maxiter=1000,
                                 workers=31,
                                 updating="deferred",
                                 init='sobol',
                                 popsize=100,
                                 tol=0.0001,
                                 disp=True)   # the random seed

    params = res.x

    np.savez("params/N2", params=params)