import sys
sys.path.append("../gym/")
from rod import Rod

import matplotlib.pyplot as plt

import numpy as np

import numpy as np 
from scipy.optimize import differential_evolution

def cost_fun(params, q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_force_save, display=False):

    params = np.power(10, params)
    rod = Rod(params[:-1])

    if params[1] > params[0]:
        return 1e2

    norm_mocap = 0
    norm_ati = 0

    cost_mocap = 0
    cost_ati = 0

    for i in range(q0_save.shape[0]):

        q0 = q0_save[i, :]
        qf = qf_save[i, :]
        traj_robot_tool = traj_robot_tool_save[i, 56:556, :]
        traj_rope_base = traj_rope_base_save[i, 56:556, :]
        # print(traj_force_save.shape)
        traj_force = traj_force_save[i, round(params[-1]):round(params[-1] + 500), :]
        
        success, traj_pos_sim, traj_force_sim_nonintertial, traj_force_sim_intertial, q_save, _ = rod.run_sim(q0, qf)

        cost_mocap += np.linalg.norm(traj_rope_base - traj_pos_sim) 
        norm_mocap += np.linalg.norm(traj_rope_base) 
        cost_ati += np.linalg.norm(traj_force - traj_force_sim_nonintertial)
        norm_ati += np.linalg.norm(traj_force)       

        if not success:
            print(1e2)
            return 1e2

        if display:
            rod.render(q_save, traj_rope_base, traj_pos_sim)

            plt.figure("pose sim v real world data")
            plt.plot(traj_rope_base, 'r-', label='real world')
            plt.plot(traj_pos_sim, 'b-', label='sim')
            plt.legend()

            plt.figure("force sim v real world data")
            plt.plot(traj_force, 'r-', label='real world')
            plt.plot(traj_force_sim_nonintertial, 'b-', label='sim noninertial')
            # plt.plot(traj_force_sim_intertial, 'g-', label='sim intertial')
            plt.legend()

            plt.show()

    # cost = cost_mocap / norm_mocap + cost_ati / norm_ati
    # cost = cost_mocap / norm_mocap
    cost = cost_ati / norm_ati

    print(cost)

    return cost

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

    params = [.254, .2413, 0.003, 0.03]
    params = np.log10(np.array(params))
    # print(cost_fun(params, q0_save, qf_save, traj_pos_save, traj_force_save, True))

    bounds = [(0.2, 0.3), (0.2, 0.3), (1e-6, 10), (0.028, 0.1), (0.1, 200)]
    bounds = np.log10(bounds)
    res = differential_evolution(cost_fun, args=[q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_force_save],                # the function to minimize
                                 bounds=bounds,
                                 maxiter=10,
                                 workers=31,
                                 updating="deferred",
                                 init='sobol',
                                 popsize=100,
                                 tol=0.0001,
                                 disp=True)   # the random seed

    params = res.x
    params = np.power(10, params)

    np.savez("params/inertial_calibration", params=params)