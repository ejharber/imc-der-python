import sys
sys.path.append("../gym/")
from rope import Rope

import matplotlib.pyplot as plt

import numpy as np

import numpy as np 
from scipy.optimize import differential_evolution

def cost_fun(params, q0_save, qf_save, traj_pos_save, traj_force_save, display=False):

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
            return 1e8

        cost_mocap += np.linalg.norm(traj_pos - traj_pos_sim) 
        norm_mocap += np.linalg.norm(traj_pos) 
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

    cost = cost_mocap / norm_mocap + cost_ati / norm_ati
    # cost = cost_mocap / norm_mocap
    # cost = cost_ati / norm_ati

    print(cost)

    return cost

if __name__ == "__main__":

    # params = [0.05, 0.05, 0.05, 1e1, 1e-2, 1e5, 1e7, 0.15, 0.01, 0.01, 0.01]
    # print(cost_fun(params, True))

    folder_name = "filtered_data"
    file = "2.npz"
    file_name = folder_name + "/" + file

    data = np.load(file_name)

    q0_save = data["q0_save"]
    qf_save = data["qf_save"]
    traj_pos_save = data["traj_pos_save"]
    traj_force_save = data["traj_force_save"]

    res = differential_evolution(cost_fun, args=[q0_save, qf_save, traj_pos_save, traj_force_save],                # the function to minimize
                                 bounds=[(1e-5, 1e-1), (1e-5, 1e-1), (1e-5, 1e-1), (1e-8, 1e4), (1e-8, 1e4), (1e1, 1e7), (1e1, 1e7), (1e1, 1e7), (1e-5, 1e1), (1e-5, 1e-1), (1e-5, 1e-1), (1e-5, 1e-1)],
                                 maxiter=2000,
                                 workers=32,
                                 updating="deferred",
                                 init='sobol',
                                 popsize=100,
                                 tol=0.00001,
                                 disp=True)   # the random seed

    print(res)

    np.savez("params/params_compression", x=res.x)
    # np.savez("res_pose", x=res.x)
    # np.savez("res_all_20", x=res.x)
