import sys
sys.path.append("../gym/")
from rope import Rope

import matplotlib.pyplot as plt

import numpy as np

import numpy as np 
from scipy.optimize import differential_evolution

def cost_fun(params, q0_save, qf_save, traj_pos_save, traj_force_save, display=False):

    params = np.power(10, params)
    rope = Rope(params)

    # q0_save = data["q0_save"]
    # qf_save = data["qf_save"]
    # traj_pos_save = data["traj_pos_save"]
    # traj_force_save = data["traj_force_save"]

    norm_mocap = 0
    norm_ati = 0

    cost_mocap = 0
    cost_ati = 0

    for i in range(q0_save.shape[0]):

        q0 = q0_save[i, :]
        qf = qf_save[i, :]
        traj_pos = traj_pos_save[i, :, :]
        traj_force = traj_force_save[i, :, :]
        
        success, traj_pos_sim, traj_force_sim, q_save, _ = rope.run_sim(q0, qf)

        if not success:
            return 1e1

        # print(traj_force_sim)
        cost_mocap += np.linalg.norm(traj_pos - traj_pos_sim) 
        norm_mocap += np.linalg.norm(traj_pos) 
        # print(traj_force.shape, traj_force_sim.shape)
        cost_ati += np.linalg.norm(traj_force - traj_force_sim)
        norm_ati += np.linalg.norm(traj_force)       

        if display:
            rope.render(q_save, traj_pos, traj_pos_sim)

            plt.figure("pose sim v real world data")
            plt.plot(traj_pos, 'r-')
            plt.plot(traj_pos_sim, 'b-')

            plt.figure("force sim v real world data")
            plt.plot(traj_force, 'r-')
            plt.plot(traj_force_sim, 'b-')

            plt.show()

    # cost = cost_mocap / norm_mocap + cost_ati / norm_ati
    # cost = cost_mocap / norm_mocap
    cost = cost_ati / norm_ati

    print(cost)

    return cost

if __name__ == "__main__":

    folder_name = "filtered_data"
    file = "intertial_tests.npz"
    file_name = folder_name + "/" + file

    data = np.load(file_name)

    q0_save = data["q0_save"]
    qf_save = data["qf_save"]
    traj_pos_save = data["traj_pos_save"]
    traj_force_save = data["traj_force_save"]

    params = [0.005, 0.005, 0.05, 1e1, 1e-2, 1e5, 1e7, 0.01, 0.01, 0.005, 0.03]
    params = np.log10(np.array(params))
    # print(cost_fun(params, q0_save, qf_save, traj_pos_save, traj_force_save, True))

    bounds = [(1e-3, 0.05), (1e-3, 0.05), (0.03, 0.05), (1e-2, 1e3), (1e-2, 1e3), (1e1, 1e8), (1e1, 1e8), (1e-5, 1), (1e-3, 1e-1), (1e-5, 1e-2), (1e-3, 1e-1)]
    bounds = np.log10(bounds)
    res = differential_evolution(cost_fun, args=[q0_save, qf_save, traj_pos_save, traj_force_save],                # the function to minimize
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