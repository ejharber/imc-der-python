import sys
sys.path.append("../gym/")

from rope_gym import RopeEnv
import numpy as np 
from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt
import dill
from skopt import dump, load

x_iters = []
cost_ee = []

def cost_fun(params):
        global x_iters
        global cost_ee

        data = np.load("data/training.npz")
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
        cost_ee_ = 0

        ## could probably speed this up using paraellization 
        # for i, action in enumerate(actions):
        #     env.reset()
        #     success, potential_training_state = env.rope.step(action)

        #     if not success:
        #     	return 1e8

        #     cost += np.linalg.norm(training_states[i] - potential_training_state)

        # for _ in range(100):
        #     print(len(actions))
        #     i = np.random.randint(low = 0, high = len(actions))
        #     env.reset()
        #     success, potential_training_state = env.rope.step(actions[i])

        #     if not success:
        #       return 1e8

        #     cost += np.linalg.norm(training_states[i] - potential_training_state)

        for _ in range(500):

            while(True):
                i = np.random.randint(low = 0, high = len(actions))
                env.reset()
                success, potential_training_state = env.rope.step(actions[i])

                if success:
                    cost += np.linalg.norm(training_states[i][:2020] - potential_training_state[:2020]) / 3
                    # cost += np.linalg.norm(training_states[i][:2020] - potential_training_state[:2020]) / 3 / 2
                    # cost += np.linalg.norm(training_states[i][2020:] - potential_training_state[2020:]) / 165 / 2
                    cost_ee_ += np.linalg.norm(training_states[i][2018:2020] - potential_training_state[2018:2020])
                    break

        x_iters.append(params)
        cost_ee.append(cost_ee_)

        print(cost_ee)

        return cost

data = np.load("data/training.npz")
actions = data["actions"]
training_states = data["training_states"]

print(cost_fun([1e-2, 1e7, 0.15, 0.2]))

# cost_fun = lambda x : cost(x, actions, training_states)

res = differential_evolution(cost_fun,                  # the function to minimize
                             [(5e-3, 1e-1), (1.9e6, 2.1e7), (1e-1, 2e-1), (1e-1, 5e-1)],
                             popsize=10,
                             maxiter=24)   # the random seed


save_file = "DE_no_force_scaled_500samples" 

np.savez(save_file, x_iters=x_iters, cost_ee=cost_ee)
dump(res, save_file + '.pkl')
