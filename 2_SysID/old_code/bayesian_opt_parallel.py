from rope_gym import RopeEnv
import numpy as np 
from skopt import gp_minimize
import matplotlib.pyplot as plt
from multiprocessing import Process, Array

def cost(params, actions, training_states):
    def simulate(EI, EA, damp, m, action, training_state, costs, i):
        env = RopeEnv()
        env.rope.EI = EI
        env.rope.EA = EA
        env.rope.damp = damp
        env.rope.m = m
        env.reset()
        success, potential_training_state = env.rope.step(action)

        if not success:
            costs[i] = 1e8
            return

        costs[i] = np.linalg.norm(training_state - potential_training_state)

    EI = params[0] 
    EA = params[1]
    damp = params[2]
    m = params[3]

    costs = Array('d', [0.0 for i in range(training_states.shape[0])])

    processes = []

    for i, action in enumerate(actions):
        p = Process(target=simulate, args=(EI, EA, damp, m, actions[i], training_states[i], costs, i))
        processes.append(p)
        processes[-1].start()

    for process in processes:
        process.join()

    cost = np.sum(costs)

    return cost

data = np.load("bayes_opt_training.npz")
actions = data["actions"]
training_states = data["training_states"]


print(cost([1e-2, 1e7, 0.15, 0.2], actions, training_states))

cost_fun = lambda x : cost(x, actions, training_states)

res = gp_minimize(cost_fun,                  # the function to minimize
                  [(1e-4, 1e-1), (1e6, 1e8), (1e-1, 1), (1e-1, 1)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=200,         # the number of evaluations of f
                  n_initial_points=20,
                  n_restarts_optimizer=20,
                  verbose=True,
                  n_jobs=10)   # the random seed


print(res)

from skopt.plots import plot_convergence
plot_convergence(res);

plt.show()