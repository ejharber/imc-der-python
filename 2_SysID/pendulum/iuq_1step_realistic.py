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

global i

def MCCostFun(x):
    global i
    return cost_fun_noise_2(x, seed=0, seed_=i)

min_x_dist = []

for i in range(1_000):
    out = differential_evolution(MCCostFun, 
        bounds = [(1e-6, 3) for _ in range(3)],    # the bounds on each dimension of x
        init='sobol',
        maxiter=20,
        # workers=4
        # n_calls=200,       # the number of evaluations of f
        )   # the random seed)

    min_x_dist.append(out.x)
    print(out.fun)
    print(i, out.x)

    min_x_dist.append(out.x)

    np.savez("IUQ_1step_realistic", min_x_dist=min_x_dist)