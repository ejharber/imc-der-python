import numpy as np

from CustomRobots import *
import seaborn as sns
from matplotlib import pyplot as plt, patches


UR5e = UR5eCustom()

q0 = [180, -80.55, 138.72, -148.02, -90, 0]
qf = [180, -100, 100, -180, -90, 0]

traj = UR5e.create_trajectory(q0, qf, time=1)
traj = UR5e.fk_traj(traj, True)
print(traj.shape)

plt.plot(traj[:, 2])
plt.show()