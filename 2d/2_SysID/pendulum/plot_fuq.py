from pendulum import *

import math
import numpy as np
import matplotlib.pyplot as plt

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

x_poses = []
y_poses = []
for i in range(10_000):
    x_measured, y_measured, success = pendulum_real(seed = i)
    x_poses.append(x_measured)
    y_poses.append(y_measured)

x_poses = np.array(x_poses).flatten()
y_poses = np.array(y_poses).flatten()

plt.figure('distribution', facecolor='white')
plt.grid(False)
plt.hist2d(x_poses, y_poses, bins=100, density=True, cmap=plt.cm.Greys, range=[[-1.1, 1.1], [-1.1, 1.1]])
cbar = plt.colorbar()
cbar.set_label('trajectory location probability')

# generate ground truth trajectory
x_sim, y_sim, success = pendulum_sim()

plt.plot(x_sim, y_sim, 'b.-', label="ground truth trajectory")

plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.title('Pendulum trajectory (over 1s)')
plt.axis('square')
plt.legend()
plt.tight_layout()

x_poses = []
y_poses = []
for i in range(10_000):
    x_measured, y_measured, success = pendulum_real_2(seed=0, seed_=i)
    x_poses.append(x_measured)
    y_poses.append(y_measured)

x_poses = np.array(x_poses).flatten()
y_poses = np.array(y_poses).flatten()

plt.figure('distribution realistic', facecolor='white')
plt.grid(False)
plt.hist2d(x_poses, y_poses, bins=100, density=True, cmap=plt.cm.Greys, range=[[-1.1, 1.1], [-1.1, 1.1]])
cbar = plt.colorbar()
cbar.set_label('trajectory location probability')

# generate ground truth trajectory
x_sim, y_sim, success = pendulum_sim()

plt.plot(x_sim, y_sim, 'b.-', label="ground truth trajectory")
x_measured, y_measured, success = pendulum_real(seed=0)
plt.plot(x_measured, y_measured, 'r.', label="measured trajectory")

plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.title('Pendulum trajectory (over 1s)')
plt.axis('square')
plt.legend()
plt.tight_layout()
plt.show()

plt.show()