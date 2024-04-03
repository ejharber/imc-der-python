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

def MCCostFun(std, n=5_000, plot=False):
    global cost_std
    global costs
    def monteCarloSample(std, n):
        global min_x
        global index
        samples = np.ones((3, n)) 
        for i in range(3): samples[i, :] *= min_x[i]
        # samples[index, :] = np.random.gumbel(min_x[index], std, n)
        # samples[index, :] = np.random.normal(min_x[index], std, n)
        # samples[index, :] = np.random.lognormal(min_x[index], std, n)
        # samples[index, :] = np.random.gamma(min_x[index]/std, std, n)
        # samples[index, :] = skewnorm.rvs(a=std[0], loc=min_x[index], scale=std[1], size=n)
        # samples[index, :] = np.random.uniform(min_x[index]-std[0], min_x[index]+std[1], n)
        samples[index, :] = np.random.triangular(min_x[index]-std[0], min_x[index], min_x[index]+std[1], n)

        I = np.where(samples[index, :] < 1e-6)
        samples[index, I[0]] = 1e-6

        costs_ = np.zeros(n)
        for i in range(n):
            costs_[i] = cost_fun(samples[:, i])

        if plot:
            plt.hist(costs_, bins=40)
            # plt.show()

        return energy_distance(costs_, costs)
        return wasserstein_distance(costs_, costs)

    return monteCarloSample(std, n)

out = differential_evolution(cost_fun, 
                     bounds = [(1e-6, 2) for _ in range(3)],    # the bounds on each dimension of x
                     init='sobol',
                     # n_calls=200,       # the number of evaluations of f
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
global cost_std
global costs

min_x = out.x

# foward uncertainty propagation
costs = []
for _ in range(5_000):
    costs.append(cost_fun_noise(min_x))

cost_std = np.std(costs)

print("give these results we can estimate we can propagate the uncertainty of our mesuremnts")
print("Cost: " + str(out.fun) + " uncertainty: " + str(cost_std))

min_x_std = []
for index in range(3):
    # out = differential_evolution(MCCostFun, 
    #                      bounds = [(1e-6, 1)],    # the bounds on each dimension of x
    #                      init='sobol',
    #                      maxiter=10,
    #                      # n_calls=200,       # the number of evaluations of f
    #                      # n_random_starts=2**6,  # the number of random initialization points
    #                      workers = 8)   # the random seed)

    out = differential_evolution(MCCostFun, 
                         bounds = [(1e-6, 1-1e-6), (1e-6, 1-1e-6)],    # the bounds on each dimension of x
                         init='sobol',
                         maxiter=100,
                         # n_calls=200,       # the number of evaluations of f
                         # n_random_starts=2**6,  # the number of random initialization points
                         )   # the random seed)

    # out = differential_evolution(MCCostFun, 
    #                      bounds = [(-10, 10), (0, 1)],    # the bounds on each dimension of x
    #                      init='sobol',
    #                      maxiter=100,
    #                      # n_calls=200,       # the number of evaluations of f
    #                      # n_random_starts=2**6,  # the number of random initialization points
    #                      )   # the random seed)

    min_x_std.append(out.x)
    print(out.fun)
    print(out.x)

    print(MCCostFun(out.x, plot=True))

    plt.show()

# print("The Length of the pendulum (L) is: " + str(round(min_x[0], 3)) + ' ± ' + str(round(min_x_std[0], 3)) + '\n' + 
#       "The dampening of the pendulum (b) is: " + str(round(min_x[1], 3)) + ' ± ' + str(round(min_x_std[1], 3)) + '\n' + 
#       "The mass of the pendulum (m) is: " + str(round(min_x[2], 3)) + '± ' + str(round(min_x_std[2], 3)))

np.savez("DE_triangular_e", min_x=min_x, min_x_std=min_x_std)

