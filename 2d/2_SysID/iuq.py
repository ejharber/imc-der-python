from pendulum import *

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from skopt import gp_minimize
from skopt.plots import plot_gaussian_process

# load data set of pendulum trajectory:
x_measured, y_measured = pendulum_real()

# estimate model parameters (system identification or sysid)
fun = lambda x: cost_fun(x, x_measured, y_measured)

# minimize the cost function 
out = gp_minimize(fun,                  # the function to minimize
                  [(1e-6, 2) for _ in range(3)],      # the bounds on each dimension of x
                  initial_point_generator='sobol',
                  n_calls=200,         # the number of evaluations of f
                  n_random_starts=2**6,  # the number of random initialization points
                  n_restarts_optimizer=10,
                  random_state=1234,
                  n_jobs = 8,
                  verbose=True)   # the random seed

print(out.models[-1])
print(out.models[-1].kernel_.get_params(deep=True))
# generate ground truth trajectory
x_sim, y_sim = pendulum_sim()

# generate trajectory as estimated by sysid
x_sysid, y_sysid = pendulum_sim(out.x[0], out.x[1], out.x[2])

print("The Length of the pendulum (L) is: " + str(out.x[0]) + '\n' + 
      "The dampening of the pendulum (b) is: " + str(out.x[1]) + '\n' + 
      "The mass of the pendulum (m) is: " + str(out.x[2]))

mean, goal_std = out.models[-1].predict(out.space.transform([out.x]), return_std=True) 
# y might be normalized too!

# for i in out.models[-1]:
    # print(i)
mean = mean[0]
goal_std = goal_std[0]
# goal_std = np.sqrt(2*len(x_measured))*0.05

# print(mean, goal_std)
# print(mean, goal_std)
# print(mean, goal_std)

# exit()

def monteCarloSample(min_x, index = 0, std = 0.01, n = 100):

    samples = np.ones((3, n)) 
    samples[0, :] *= min_x[0]
    samples[1, :] *= min_x[1]
    samples[2, :] *= min_x[2]
    samples[index, :] = np.random.normal(min_x[index], std, n)
    I = np.where(samples[index, :] < 1e-4)
    samples[index, I[0]] = 1e-4
    # print(I[0])
    # exit()
    costs = np.zeros(n)
    for i in range(n):
        costs[i] = fun(samples[:, i])

    # plt.figure(1)
    # plt.hist(samples[0, :], bins=40)

    # plt.figure(2)
    # plt.hist(samples[1, :], bins=40)

    # plt.figure(3)
    # plt.hist(samples[2, :], bins=40)

    # plt.figure(4)
    # plt.hist(costs, bins=40)
    # plt.show()

    return np.std(costs)

# monteCarloSample(out.x, 0, n=10000)
# monteCarloSample(out.x, 1, n=1000)
# monteCarloSample(out.x, 2, n=1000)

# exit()

def MCCost(min_x, index, std, goal_std, n):
    cost = monteCarloSample(np.copy(min_x), index, std[0], n) 
    # print(std, cost, goal_std)
    cost -= goal_std
    cost *= cost
    return cost

fun_mc = lambda X: MCCost(out.x, 0, X, goal_std, 1000)

out_0 = gp_minimize(fun_mc,                  # the function to minimize
                  [(1e-4, 1)],      # the bounds on each dimension of x
                  # acq_func="EI",      # the acquisition function
                  n_calls=100,         # the number of evaluations of f
                  n_random_starts=20,  # the number of random initialization points
                  # noise=0.01,       # the noise level (optional)
                  random_state=1234,
                  n_jobs = 8,
                  verbose=False)   # the random seed

# print(fun_mc(out_0.x))
# print(out_0.x)

fun_mc = lambda X: MCCost(out.x, 1, X, goal_std, 1000)

out_1 = gp_minimize(fun_mc,                  # the function to minimize
                  [(1e-4, 1)],      # the bounds on each dimension of x
                  # acq_func="EI",      # the acquisition function
                  n_calls=100,         # the number of evaluations of f
                  n_random_starts=20,  # the number of random initialization points
                  # noise=0.01,       # the noise level (optional)
                  random_state=1234,
                  n_jobs = 8,
                  verbose=False)   # the random seed

# print(fun_mc(out_1.x))
# print(out_1.x)
# print(out_1.models[-1])
# print(out_1.models[-1].kernel_.get_params(deep=True))
# exit()

fun_mc = lambda X: MCCost(out.x, 2, X, goal_std, 1000)

out_2 = gp_minimize(fun_mc,                  # the function to minimize
                  [(1e-4, 1)],      # the bounds on each dimension of x
                  # acq_func="EI",      # the acquisition function
                  n_calls=100,         # the number of evaluations of f
                  n_random_starts=20,  # the number of random initialization points
                  # noise=0.01,       # the noise level (optional)
                  random_state=1234,
                  n_jobs = 8,
                  verbose=False)   # the random seed

print(fun_mc(out_2.x))
print(out_2.x)

print("The Length of the pendulum (L) is: " + str(round(out.x[0], 3)) + ' ± ' + str(round(out_0.x[0], 3)) + '\n' + 
      "The dampening of the pendulum (b) is: " + str(round(out.x[1], 3)) + ' ± ' + str(round(out_1.x[0], 3)) + '\n' + 
      "The mass of the pendulum (m) is: " + str(round(out.x[2], 3)) + '± ' + str(round(out_2.x[0], 3)))
