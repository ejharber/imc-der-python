from pendulum import *

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from skopt import gp_minimize
from skopt import forest_minimize
from scipy.optimize import differential_evolution

def MCCostFun(std, n=1000):
    global cost_std
    def monteCarloSample(std, n):
        global min_x
        global index
        samples = np.ones((3, n)) 
        for i in range(3): samples[i, :] *= min_x[i]
        samples[index, :] = np.random.normal(min_x[index], std, n)
        I = np.where(samples[index, :] < 1e-4)
        samples[index, I[0]] = 1e-4
        costs = np.zeros(n)
        for i in range(n):
            costs[i] = cost_fun(samples[:, i])

        return np.std(costs)

    cost = (cost_std - monteCarloSample(std[0], n))**2
    return cost

out = gp_minimize(cost_fun_noise,                  # the function to minimize
                  [(1e-6, 2) for _ in range(3)],      # the bounds on each dimension of x
                  initial_point_generator='sobol',
                  n_calls=200,         # the number of evaluations of f
                  n_random_starts=2**6,  # the number of random initialization points
                  n_restarts_optimizer=10,
                  n_jobs = 8,
                  verbose=False)   # the random seed

print("BO")
pred_mean, pred_std = out.models[-1].predict(out.space.transform([out.x]), return_std=True) 
print("predicted minimum: " + str(out.fun) + " uncertainty: " + str(pred_std))
print("minimum: " + str(cost_fun(out.x)))

print("The Length of the pendulum (L) is: " + str(out.x[0]) + '\n' + 
    "The dampening of the pendulum (b) is: " + str(out.x[1]) + '\n' + 
    "The mass of the pendulum (m) is: " + str(out.x[2]))

global min_x
global index
global cost_std

min_x = out.x

cost_std = pred_std[0]

print("give these results we can estimate we can propagate the uncertainty of our mesuremnts")
print("Cost: " + str(out.fun) + " uncertainty: " + str(cost_std))

min_x_std = []
for index in range(3):
    out = gp_minimize(MCCostFun,                  # the function to minimize
                  [(1e-6, 1)],      # the bounds on each dimension of x
                  initial_point_generator='sobol',
                  n_calls=200,         # the number of evaluations of f
                  n_random_starts=2**6,  # the number of random initialization points
                  n_restarts_optimizer=10,
                  n_jobs = 8,
                  verbose=False)   # the random seed

    min_x_std.append(out.x[0])
    print(out.fun)
    print(out.x[0])

print("The Length of the pendulum (L) is: " + str(round(min_x[0], 3)) + ' ± ' + str(round(min_x_std[0], 3)) + '\n' + 
      "The dampening of the pendulum (b) is: " + str(round(min_x[1], 3)) + ' ± ' + str(round(min_x_std[1], 3)) + '\n' + 
      "The mass of the pendulum (m) is: " + str(round(min_x[2], 3)) + '± ' + str(round(min_x_std[2], 3)))
