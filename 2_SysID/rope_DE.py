import sys
sys.path.append("../gym/")
from rope import Rope

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from concurrent.futures import ProcessPoolExecutor, as_completed

def compute_cost(params, q0_save, qf_save, traj_rope_tip_save, traj_force_save, display=False, callback=False):
    cost_mocap_x = 0
    cost_mocap_y = 0
    cost_ati = 0

    print(params.shape)
    params = np.power(10, params)

    for i in range(q0_save.shape[0]):
        q0 = q0_save[i, :]
        qf = qf_save[i, :]

        traj_rope_tip = traj_rope_tip_save[i, round(params[-1]):round(params[-1] + 500), :]
        traj_force = traj_force_save[i, round(params[-2]):round(params[-2] + 500), :]

        rope = Rope(params[:-2])
        success, traj_pos_sim, traj_force_sim, traj_force_sim_base, traj_force_sim_rope, q_save, _ = rope.run_sim(q0, qf)

        if not success:
            print()
            if callback:
                return [1e4, 1e4, 1e4, 1e4]  # Large cost if simulation fails
            else: return 1e4

        # Calculate costs for this trial
        cost_mocap_x += np.linalg.norm(traj_rope_tip[:, 0] - traj_pos_sim[:, 0])
        cost_mocap_y += np.linalg.norm(traj_rope_tip[:, 1] - traj_pos_sim[:, 1])
        cost_ati += np.linalg.norm(traj_force - traj_force_sim)

    cost_mocap_x /= (np.max(traj_rope_tip_save[:, :, 0]) - np.min(traj_rope_tip_save[:, :, 0]))
    cost_mocap_y /= (np.max(traj_rope_tip_save[:, :, 1]) - np.min(traj_rope_tip_save[:, :, 1]))
    cost_ati /= (np.max(traj_force_save) - np.min(traj_force_save))

    total_cost = cost_mocap_x + cost_mocap_y + 2 * cost_ati

    print(total_cost)
    print(params)

    if callback:
        return [total_cost, cost_mocap_x, cost_mocap_y, cost_ati]
    else: return total_cost

# Cost function with parallel evaluation
def cost_fun(param_sets, q0_save, qf_save, traj_rope_tip_save, traj_force_save, display=False, callback=False):
    """
    Vectorized cost function to handle multiple parameter sets in parallel.
    Each param set is evaluated in a separate process using ProcessPoolExecutor.
    """

    param_sets = np.atleast_2d(param_sets).T  # Ensure it's 2D if single parameter set is passed

    if param_sets.shape[0] == 15:
        param_sets = param_sets.T

    # List to store the results for all parameter sets along with their indices
    indexed_results = []

    # Multithreading to compute the cost for each parameter set
    with ProcessPoolExecutor() as executor:
        # Submit tasks for each set of parameters, passing the index along
        futures = {executor.submit(compute_cost, params, q0_save, qf_save, traj_rope_tip_save, traj_force_save, display, callback): idx for idx, params in enumerate(param_sets)}

        # Gather the results as they complete
        for future in as_completed(futures):
            idx = futures[future]  # Retrieve the index of the current task
            try:
                result = future.result()
            except Exception as e:
                print(f"Error in process: {e}")
                if callback:
                    result = [1e4, 1e4, 1e4, 1e4]  # Assign large cost in case of failure
                else:
                    return 1e4

            indexed_results.append((idx, result))

    # Sort the results by the original index to maintain order
    indexed_results.sort(key=lambda x: x[0])

    # Extract just the results, discarding the indices
    results = [res for _, res in indexed_results]

    # Return the results as a numpy array (expected shape for vectorize=True)
    return np.array(results)

# Class to store cost history
class SaveCostHistory:
    def __init__(self):
        self.total_costs = []
        self.mocap_x_costs = []
        self.mocap_y_costs = []
        self.ati_costs = []

    def __call__(self, params, convergence):
        costs = cost_fun([params], q0_save, qf_save, traj_rope_tip_save, traj_force_save, callback=True)[0]
        print(costs.shape)
        total_cost, mocap_x_cost, mocap_y_cost, ati_cost = costs
        self.total_costs.append(total_cost)
        self.mocap_x_costs.append(mocap_x_cost)
        self.mocap_y_costs.append(mocap_y_cost)
        self.ati_costs.append(ati_cost)

def cacluate_std(params, q0_save, qf_save, traj_rope_tip_save, traj_force_save):

    error_mocap_x = []
    error_mocap_y = []
    error_ati = []

    print(params.shape)

    for i in range(q0_save.shape[0]):
        q0 = q0_save[i, :]
        qf = qf_save[i, :]

        traj_rope_tip = traj_rope_tip_save[i, round(params[-1]):round(params[-1] + 500), :]
        traj_force = traj_force_save[i, round(params[-2]):round(params[-2] + 500), :]

        rope = Rope(params[:-2])
        success, traj_pos_sim, traj_force_sim, _, traj_force_sim_rope, q_save, _ = rope.run_sim(q0, qf)

        # Calculate costs for this trial
        error_mocap_x.append(traj_rope_tip[:, 0] - traj_pos_sim[:, 0])
        error_mocap_y.append(traj_rope_tip[:, 1] - traj_pos_sim[:, 1])
        error_ati.append(traj_force - traj_force_sim)

    error_mocap_x = np.array(error_mocap_x)
    error_mocap_y = np.array(error_mocap_y)
    error_ati = np.array(error_ati)

    std_x = np.std(error_mocap_x)
    std_y = np.std(error_mocap_y)
    std_ati = np.std(error_ati)

    error_mocap_x = np.linalg.norm(error_mocap_x)
    error_mocap_y = np.linalg.norm(error_mocap_y)
    error_ati = np.linalg.norm(error_ati)

    return error_mocap_x, error_mocap_y, error_ati, std_x, std_y, std_ati

if __name__ == "__main__":
    folder_name = "filtered_data"
    file = "N2.npz"
    file_name = folder_name + "/" + file
    data = np.load(file_name)

    traj_rope_tip_save = data["traj_rope_tip_save"]
    traj_force_save = data["traj_force_save"]
    q0_save = data["q0_save"]
    qf_save = data["qf_save"]

    bounds = [(0.0001, 0.4), (0.0001, 0.2), (0.4, 0.8),  # DLs
              (0.001, 1e2), (1, 1e4),  # Kb
              (1e1, 1e5), (1e1, 1e5),  # Ks
              (1e-4, .1), (1e-5, 1e2), (1e-5, 1e2),  # damping
              (0.028, 0.2), (.0103 - 0.01, .0103 + 0.02), (.07380 - 0.01, .07380 + 0.01),  # mass
              (500, 600), (500, 600)]  # time sync

    log_bounds = [(np.log10(lower), np.log10(upper)) for lower, upper in bounds if lower > 0 and upper > 0]

    # Initialize cost history tracker
    save_cost_history = SaveCostHistory()

    # Run differential evolution optimization with `vectorize=True`
    res = differential_evolution(
        cost_fun, 
        args=[q0_save, qf_save, traj_rope_tip_save, traj_force_save],  
        bounds=log_bounds,
        maxiter=100,
        # polish=False,
        # popsize=1,
        updating="deferred",
        init='sobol',
        vectorized=True,  # Enables vectorized mode
        tol=0.001,
        disp=True,
        callback=save_cost_history  # Track cost over iterations
    )

    params = res.x
    params = np.power(10, params)

    # Calculate standard deviation
    error_mocap_x, error_mocap_y, error_ati, std_x, std_y, std_ati = cacluate_std(params, q0_save, qf_save, traj_rope_tip_save, traj_force_save)

    np.savez("params/N2_all", params=params, costs=save_cost_history.total_costs, mocap_x_costs=save_cost_history.mocap_x_costs,
             mocap_y_costs=save_cost_history.mocap_y_costs, ati_costs=save_cost_history.ati_costs, 
             error_mocap_x=error_mocap_x, error_mocap_y=error_mocap_y, error_ati=error_ati, 
             std_x=std_x, std_y=std_y, std_ati=std_ati)

    # Plot cost history over iterations
    plt.plot(save_cost_history.total_costs, label='Total Cost', marker='o')
    plt.plot(save_cost_history.mocap_x_costs, label='Mocap X Cost', marker='x')
    plt.plot(save_cost_history.mocap_y_costs, label='Mocap Y Cost', marker='x')
    plt.plot(save_cost_history.ati_costs, label='ATI Cost', marker='x')
    plt.title('Cost Function Components Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.show()
