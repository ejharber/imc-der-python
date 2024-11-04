import sys
sys.path.append("../gym/")
from rope import Rope

import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.expx import ExponentialCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.evaluator import Evaluator

# Add multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pymoo.core.parallelization import Parallelization


# Cost function with simulation and computation logic
def cost_fun(params, q0_save, qf_save, traj_robot_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save, display=False):
    cost_mocap_x = 0
    cost_mocap_y = 0
    cost_ati = 0

    params = np.power(10, params)

    for i in range(q0_save.shape[0]):
        q0 = q0_save[i, :]
        qf = qf_save[i, :]

        traj_robot = traj_robot_save[i, :, :]
        traj_rope_base = traj_rope_base_save[i, round(params[-1]):round(params[-1] + 500), :]
        traj_rope_tip = traj_rope_tip_save[i, round(params[-1]):round(params[-1] + 500), :]
        traj_force = traj_force_save[i, round(params[-2]):round(params[-2] + 500), :]

        rope = Rope(params[:-2])
        success, traj_pos_sim, traj_force_sim, traj_force_sim_base, traj_force_sim_rope, q_save, _ = rope.run_sim(q0, qf)

        if not success:
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
           2 * cost_ati / (np.max(traj_force_save) - np.min(traj_force_save)))

    print(cost)
    print(repr(params))
    print()

    return cost


# Class to store cost history
class SaveCostHistory(Callback):
    def __init__(self):
        super().__init__()
        self.costs = []

    def notify(self, algorithm):
        cost = cost_fun(
            algorithm.pop.get("X")[-1], q0_save, qf_save,
            traj_robot_tool_save, traj_rope_base_save,
            traj_rope_tip_save, traj_force_save
        )
        self.costs.append(cost)


# Problem definition
class RopeOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=len(bounds), n_obj=1, xl=log_lower_bounds, xu=log_upper_bounds)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array([cost_fun(params, q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save) for params in x])


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

    bounds = [(0.0001, 0.4), (0.0001, 0.2), (0.4, 0.8),  # DLs
              (0.001, 1e2), (1, 1e4),  # Kb
              (1e1, 1e5), (1e1, 1e5),  # Ks
              (1e-4, .1), (1e-5, 1e5), (1e-5, 1e5),  # damping
              (0.028, 0.1), (.0103 - 0.01, .0103 + 0.01), (.06880 - 0.01, .06880 + 0.01),  # mass
              (20, 100), (20, 100)]  # time sync

    log_bounds = [(np.log10(lower), np.log10(upper)) for lower, upper in bounds if lower > 0 and upper > 0]
    log_lower_bounds = [b[0] for b in log_bounds]
    log_upper_bounds = [b[1] for b in log_bounds]

    # Initialize cost history tracker
    save_cost_history = SaveCostHistory()

    # Create the problem
    problem = RopeOptimizationProblem()

    # Define the DE algorithm properly
    algorithm = DE(
        pop_size=100,
        sampling=LHS(),
        variant="DE/rand/1/exp",  # Specify the variant explicitly for exponential crossover
        mutation=PolynomialMutation(eta=20),  # Mutation remains the same
        CR=0.9, F=0.8
    )

    # Set the termination condition
    termination = get_termination("n_gen", 200)

    # Setup parallelization using ProcessPoolExecutor
    parallel = Parallelization(
        method="threading",  # You can use "threading" or "multiprocessing"
        n_jobs=-1,           # Use all available cores
        eval_type="by_thread"
    )

    # Run optimization
    res = minimize(
        problem,
        algorithm,
        termination,
        callback=save_cost_history,
        disp=True,
        seed=1,
        evaluator=parallel  # Pass the parallel evaluator
    )

    params = res.X
    params = np.power(10, params)

    std_x, std_y, std_force = cacluate_std(params, q0_save, qf_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save)

    np.savez("params/N2_axialdamp", params=params, costs=save_cost_history.costs, std_x=std_x, std_y=std_y, std_force=std_force)

    # Plot cost history over iterations
    plt.plot(save_cost_history.costs, marker='o')
    plt.title('Cost Function Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()
