import sys
sys.path.append("../gym/")
from rope import Rope

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from concurrent.futures import ProcessPoolExecutor, as_completed

class SaveCostHistory:
    def __init__(self, save_folder_name, data_file, save_interval=10):
        self.save_folder_name = save_folder_name
        self.data_file = data_file
        self.save_interval = save_interval
        self.iteration = 0
        self.total_costs = []
        self.mocap_x_costs = []
        self.mocap_x_costs_std = []
        self.mocap_y_costs = []
        self.mocap_y_costs_std = []
        self.ati_costs = []
        self.ati_costs_std = []
        self.params_history = []

    def save(self, params, final_iteration=False):
        """Save the optimization history and errors to a file."""
        file_name = (
            f"{self.save_folder_name}/{self.data_file}_final"
            if final_iteration
            else f"{self.save_folder_name}/{self.data_file}_{self.iteration}"
        )

        np.savez(file_name,
                 params=np.power(10, params),
                 total_costs=self.total_costs,
                 mocap_x_costs=self.mocap_x_costs,
                 mocap_x_costs_std=self.mocap_x_costs_std,
                 mocap_y_costs=self.mocap_y_costs,
                 mocap_y_costs_std=self.mocap_y_costs_std,
                 ati_costs=self.ati_costs,
                 ati_costs_std=self.ati_costs_std,
                 params_history = self.params_history)

    def __call__(self, params, convergence):
        """Callback function to track and save progress during optimization."""
        costs = compute_cost(params, q0_save, qf_save, traj_rope_tip_save, traj_force_save, callback=True)
        total_cost, cost_mocap_x, cost_mocap_y, cost_ati, std_x, std_y, std_ati = costs

        self.total_costs.append(total_cost)
        self.mocap_x_costs.append(cost_mocap_x)
        self.mocap_x_costs_std.append(std_x)
        self.mocap_y_costs.append(cost_mocap_y)
        self.mocap_y_costs_std.append(std_y)
        self.ati_costs.append(cost_ati)
        self.ati_costs_std.append(std_ati)
        self.params_history.append(np.power(10, params))

        self.iteration += 1
        if self.iteration % self.save_interval == 0:
            self.save(params)

def compute_cost(params, q0_save, qf_save, traj_rope_tip_save, traj_force_save, callback=False):

    params = np.power(10, params)

    error_mocap_x, error_mocap_y, error_ati = [], [], []

    for i in range(q0_save.shape[0]):
        q0 = q0_save[i, :]
        qf = qf_save[i, :]

        traj_rope_tip = traj_rope_tip_save[i, round(params[-1]):round(params[-1] + 500), :]
        traj_force = traj_force_save[i, round(params[-2]):round(params[-2] + 500), :]

        rope = Rope(params[:-2])
        success, traj_pos_sim, traj_force_sim, _, _, _, _ = rope.run_sim(q0, qf)

        if not success:
            if callback:
                return [1e4, 1e4, 1e4, 1e4, 0, 0, 0]
            else:
                return 1e4

        error_mocap_x.append(traj_rope_tip[:, 0] - traj_pos_sim[:, 0])
        error_mocap_y.append(traj_rope_tip[:, 1] - traj_pos_sim[:, 1])
        error_ati.append(traj_force - traj_force_sim)

    error_mocap_x = np.array(error_mocap_x)
    error_mocap_y = np.array(error_mocap_y)
    error_ati = np.array(error_ati)

    std_x = np.std(error_mocap_x)
    std_y = np.std(error_mocap_y)
    std_ati = np.std(error_ati)

    cost_mocap_x = np.linalg.norm(error_mocap_x) / (np.max(traj_rope_tip_save[:, :, 0]) - np.min(traj_rope_tip_save[:, :, 0]))
    cost_mocap_y = np.linalg.norm(error_mocap_y) / (np.max(traj_rope_tip_save[:, :, 1]) - np.min(traj_rope_tip_save[:, :, 1]))
    cost_ati = np.linalg.norm(error_ati) / (np.max(traj_force_save) - np.min(traj_force_save))

    total_cost = cost_mocap_x + cost_mocap_y + 2 * cost_ati

    print(total_cost)

    if callback:
        return [total_cost, cost_mocap_x, cost_mocap_y, cost_ati, std_x, std_y, std_ati]
    else:
        return total_cost

if __name__ == "__main__":

    data_folder_name = "filtered_data"
    save_folder_name = "params"
    data_file = "N2"
    data_file_name = data_folder_name + "/" + data_file + ".npz"
    save_file = "N2_all"
    data = np.load(data_file_name)
    save_interval = 10

    traj_rope_tip_save = data["traj_rope_tip_save"]
    traj_force_save = data["traj_force_save"]
    q0_save = data["q0_save"]
    qf_save = data["qf_save"]

    bounds = [(0.203, 0.228), (0.0001, 0.1), (0.5, 0.6),
              (0.001, 1e2), (1, 1e4),
              (1e1, 1e5), (1e1, 1e5),
              (1e-5, 0.1), (1e-5, 1e2), (1e-5, 1e2),
              (0.028, 0.2), (.0103 - 0.01, .0103 + 0.02), (.07380 - 0.01, .07380 + 0.01),
              (100, 120), (130, 180)]

    save_cost_history = SaveCostHistory(save_folder_name, save_file, save_interval=save_interval)

    log_bounds = [(np.log10(lower), np.log10(upper)) for lower, upper in bounds if lower > 0 and upper > 0]

    res = differential_evolution(
        compute_cost,
        args=[q0_save, qf_save, traj_rope_tip_save, traj_force_save],
        bounds=log_bounds,
        maxiter=100,
        updating="deferred",
        workers=20,
        # popsize=100,
        tol=0.01,
        disp=True,
        callback=save_cost_history
    )

    # Final save
    params = res.x
    save_cost_history(params, False)
    save_cost_history.save(params, final_iteration=True)

    plt.plot(save_cost_history.total_costs, label='Total Cost', marker='o')
    plt.title('Cost Function Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.show()
