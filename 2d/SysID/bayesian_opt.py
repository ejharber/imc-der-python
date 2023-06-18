import sys
sys.path.append("../gym/")

from rope_gym import RopeEnv
import numpy as np 
from skopt import gp_minimize
import matplotlib.pyplot as plt
import dill
from skopt import dump, load
from skopt.plots import plot_convergence

def cost_fun(params):

        data = np.load("bayes_opt_training_large.npz")
        actions = data["actions"]
        training_states = data["training_states"]

        print(params)
        EI = params[0] 
        EA = params[1]
        damp = params[2]
        m = params[3]

        env = RopeEnv()

        env.rope.EI = EI
        env.rope.EA = EA
        env.rope.damp = damp
        env.rope.m = m

        cost = 0

        for _ in range(100):

            while(True):
                i = np.random.randint(low = 0, high = len(actions))
                env.reset()
                success, potential_training_state = env.rope.step(actions[i])

                if success:
                    cost += np.linalg.norm(training_states[i][:2020] - potential_training_state[:2020]) # / 3
                    # cost += np.linalg.norm(training_states[i][:2020] - potential_training_state[:2020]) #  / 3 / 2
                    # cost += np.linalg.norm(training_states[i][2020:] - potential_training_state[2020:]) # / 165 / 2
                    break

        return cost

data = np.load("bayes_opt_training_large.npz")
actions = data["actions"]
training_states = data["training_states"]

print(cost_fun([1e-2, 1e7, 0.15, 0.2]))

# cost_fun = lambda x : cost(x, actions, training_states)

res = gp_minimize(cost_fun,                  # the function to minimize
                  [(5e-3, 1e-1), (1.9e6, 2.1e7), (1e-1, 2e-1), (1e-1, 5e-1)],      # the bounds on each dimension of x
                  n_calls=1000,         # the number of evaluations of f
                  n_initial_points=100,
                  # n_restarts_optimizer=12,
                  verbose=True,
                  noise=5e-4)   # the random seed

dump(res, 'BO_no_force.pkl')

print(res)

plot_convergence(res);

plt.show()