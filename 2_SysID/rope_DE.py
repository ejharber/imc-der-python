import sys
sys.path.append("../gym/")
from rope import Rope

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

# Cost function with simulation and computation logic
def cost_fun(params, inertial_params, q0_save, qf_save, traj_robot_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save, display=False):
    cost_mocap_x = 0
    cost_mocap_y = 0
    cost_ati = 0

    params = np.power(10, params)

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
        
        cost_mocap_x += np.linalg.norm(traj_rope_tip[:, 0] - traj_pos_sim[:, 0])
        cost_mocap_y += np.linalg.norm(traj_rope_tip[:, 1] - traj_pos_sim[:, 1])
        cost_ati += np.linalg.norm(traj_force - traj_force_sim)

        if display:
            plt.figure("force sim v real world data")
            plt.plot(traj_force, 'r.', label='real world')
            plt.plot(traj_force_sim, 'r-', label='sim total')
            plt.plot(traj_force_sim_base, 'b-', label='sim base')
            plt.plot(traj_force_sim_rope, 'g-', label='sim rope')
            plt.legend()
            plt.show()
            exit()

    cost = (cost_mocap_x / (np.max(traj_rope_tip_save[:, :, 0]) - np.min(traj_rope_tip_save[:, :, 0])) +  
           cost_mocap_y / (np.max(traj_rope_tip_save[:, :, 1]) - np.min(traj_rope_tip_save[:, :, 1])) + 
           cost_ati / (np.max(traj_force_save) - np.min(traj_force_save)))

    print(cost)
    print(repr(params))
    print()

    return cost

# Class to store cost history
class SaveCostHistory:
    def __init__(self):
        self.costs = []

    def __call__(self, params, convergence):
        cost = cost_fun(params, inertial_params, q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save)
        self.costs.append(cost)

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

    bounds = [(0.0001, 0.4), (0.0001, 0.2), (0.4, 0.8), # DLs
              (0.001, 1e2), (1, 1e4), 
              (1e1, 1e5), (1e1, 1e5), # Ks
              (1e-4, .1), (1e-5, .1), # damping
              (0.028, 0.1), (.0103 - 0.02, .0103 + 0.02), (.06880 - 0.02, .06880 + 0.02), # mass
              (20, 100), (20, 100)] # time sync

    log_bounds = [(np.log10(lower), np.log10(upper)) for lower, upper in bounds if lower > 0 and upper > 0]

    # Initialize cost history tracker
    save_cost_history = SaveCostHistory()

    # Run differential evolution optimization
    res = differential_evolution(
        cost_fun, 
        args=[inertial_params, q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save],  
        bounds=log_bounds,
        maxiter=500,
        workers=31,
        updating="deferred",
        init='sobol',
        # popsize=50,
        tol=0.0001,
        disp=True,
        callback=save_cost_history  # Track cost over iterations
    )

    params = res.x
    np.savez("params/N2", params=params, costs=save_cost_history.costs)

    # Plot cost history over iterations
    plt.plot(save_cost_history.costs, marker='o')
    plt.title('Cost Function Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()
