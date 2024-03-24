from pendulum import *

import math
import numpy as np
import matplotlib.pyplot as plt


data = np.load("DE.npz")
min_x = data["min_x"]
min_x_std = data["min_x_std"]

# foward uncertainty propagation
costs = []
for _ in range(1000):
    costs.append(cost_fun_noise(min_x))

cost_std = np.std(costs)


