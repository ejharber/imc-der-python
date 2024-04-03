from pendulum import *

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from skopt import gp_minimize
from skopt import forest_minimize
from scipy.optimize import differential_evolution
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance
from scipy.stats import skewnorm

def MCCostFun(x):
    global min_x
    global index
    global cost_goal

    min_x_ = np.copy(min_x) 
    min_x_[index] = x

    return (cost_fun(min_x_) - cost_goal) ** 2

out = differential_evolution(cost_fun, 
                     bounds = [(1e-6, 2) for _ in range(3)],    # the bounds on each dimension of x
                     init='sobol',
                     maxiter=10000,       # the number of evaluations of f
                     # n_random_starts=2**6,  # the number of random initialization points
                     workers = 8)   # the random seed)

print("DE")
print("predicted minimum: " + str(out.fun))
print("minimum: " + str(cost_fun(out.x)))

print("The Length of the pendulum (L) is: " + str(out.x[0]) + '\n' + 
    "The dampening of the pendulum (b) is: " + str(out.x[1]) + '\n' + 
    "The mass of the pendulum (m) is: " + str(out.x[2]))

global min_x
global index
global cost_goal

min_x = out.x

# foward uncertainty propagation
costs = []
for _ in range(5_000):
    costs.append(cost_fun_noise(min_x))

min_x_dist = []
for index in range(3):

    min_x_dist_ = []
    for cost_goal in costs:

        out = differential_evolution(MCCostFun, 
                             bounds = [(-min_x[index] + 1e-6, 4*min_x[index])],    # the bounds on each dimension of x
                             init='sobol',
                             maxiter=100,
                             # n_calls=200,       # the number of evaluations of f
                             )   # the random seed)

        min_x_dist_.append(out.x)
        print(out.fun)
        print(out.x)

    min_x_dist.append(min_x_dist_)

np.savez("MC_IUQ", costs=costs, min_x=min_x, min_x_dist=min_x_dist)

