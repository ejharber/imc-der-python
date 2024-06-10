from pendulum import *

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from skopt import gp_minimize
from skopt import forest_minimize
from scipy.optimize import differential_evolution

# estimate model parameters (system identification or sysid)

out = forest_minimize(cost_fun,                  # the function to minimize
                  [(1e-6, 2) for _ in range(3)],      # the bounds on each dimension of x
                  verbose=False)   # the random seed

print("Forest")
print("predicted minimum: " + str(out.fun))
print("minimum: " + str(cost_fun(out.x)))

print("The Length of the pendulum (L) is: " + str(out.x[0]) + '\n' + 
      "The dampening of the pendulum (b) is: " + str(out.x[1]) + '\n' + 
      "The mass of the pendulum (m) is: " + str(out.x[2]))

out = gp_minimize(cost_fun,                  # the function to minimize
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

out = differential_evolution(cost_fun, 
                             bounds = [(1e-6, 2) for _ in range(3)],      # the bounds on each dimension of x
                             init='sobol',
                             # n_calls=200,         # the number of evaluations of f
                             # n_random_starts=2**6,  # the number of random initialization points
                             workers = 8)   # the random seed)

print("DE")
print("predicted minimum: " + str(out.fun))
print("minimum: " + str(cost_fun(out.x)))

print("The Length of the pendulum (L) is: " + str(out.x[0]) + '\n' + 
      "The dampening of the pendulum (b) is: " + str(out.x[1]) + '\n' + 
      "The mass of the pendulum (m) is: " + str(out.x[2]))
