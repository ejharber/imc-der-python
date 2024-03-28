from pendulum import *

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

data = np.load("DE.npz")
min_x = data["min_x"]
min_x_std = data["min_x_std"]

print(min_x, min_x_std)
# x_poses = []
# y_poses = []
# for i in range(1000):
#     x_measured, y_measured = pendulum_real(min_x[0], min_x[1], min_x[2], seed = i)
#     x_poses.append(x_measured)
#     y_poses.append(y_measured)

# x_poses = np.array(x_poses).flatten()
# y_poses = np.array(y_poses).flatten()

# plt.figure('distribution', facecolor='white')
# plt.hist2d(x_poses, y_poses, bins=100, density=True, cmap=plt.cm.Greys, range=[[-1.1, 1.1], [-1.1, 1.1]])
# cbar = plt.colorbar()
# cbar.set_label('trajectory location probability')
# plt.tight_layout()

# x_sysid, y_sysid = pendulum_sim(min_x[0], min_x[1], min_x[2])
# plt.plot(x_sysid, y_sysid, 'k.-', label="sysID trajectory")
# x_sysid, y_sysid = pendulum_sim()
# plt.figure('distribution', facecolor='white')
# plt.plot(x_sysid, y_sysid, 'b.-', label="ground truth")
# # plt.set_facecolor("white")

# plt.xlabel('x position (m)')
# plt.ylabel('y position (m)')
# plt.title('Pendulum trajectory (over 1s)')
# plt.axis('square')
# plt.xlim([-1.1, 1.1])
# plt.ylim([-1.1, 1.1])

# plt.legend()


# plt.figure("cost distribution")
# # foward uncertainty propagation
# costs = []
# for _ in range(1000):
#     costs.append(cost_fun_noise(min_x))

# costs = np.array(costs)

# cost_std = np.std(costs)
# cost_mean = np.mean(costs)

# # plt.figure("Cost Uncertainty")
# plt.hist(costs, bins=40, density=True, alpha=0.8)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, cost_mean, cost_std)
# plt.plot(x, p, 'r', linewidth=2)

# plt.xlabel('cost')
# plt.ylabel('probability')

# title = "Cost Function with propagated uncertainties (Mean: {:.2f} and std: {:.2f})".format(cost_mean, cost_std)
# plt.title(title)
costs_real = []
for _ in range(5_000):
    costs_real.append(cost_fun_noise(min_x))

# plt.hist(costs_real)

# costs_real = []
# for _ in range(500):
    # costs_real.append(cost_fun_noise([1, 0.5 ,1]))

# plt.hist(costs_real)

# plt.show()
cost_std = np.std(costs_real)
print(cost_std)

for i in range(len(min_x_std)):
    costs_est = []
    for _ in range(500):
        # std = np.zeros(3)
        # std = min_x_std[i]
        min_x_sample = np.copy(min_x)
        min_x_sample[i] = np.random.lognormal(min_x[i], min_x_std[i])

        print(min_x, min_x_sample)
        # min_x_sample = np.copy(min_x) + stds
        costs_est.append(cost_fun(min_x_sample))
    
    print(np.std(costs_est), np.std(costs_real))

    plt.figure(i)
    plt.hist(costs_real, bins=40, alpha=0.8)
    # plt.plot
    plt.figure(i+1)
    plt.hist(costs_est, bins=40, alpha=0.8)
    plt.show()
plt.show()
