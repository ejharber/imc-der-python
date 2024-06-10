from rope import RopePython
import numpy as np
import matplotlib.pyplot as plt
import time

rope = RopePython(True, "Human")
rope.reset()

action = np.array([0.1, -0.1, np.pi/4]).T
success, traj_pos, traj_force = rope.step(action)

rope.close()