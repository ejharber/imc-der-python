from rope import RopePython
import numpy as np
import matplotlib.pyplot as plt
import time

max_lim = 0.2
min_lim = -0.2

max_lim_angle = 3 / 2 * np.pi
min_lim_angle = -3 / 2 * np.pi

for i in range(10):
    rope = RopePython(False, "Human")

    for _ in range(10):

        rope.reset()

        rand_action = np.random.random((3, 1))
        rand_action[:2] = rand_action[:2] * (max_lim - min_lim) + min_lim
        rand_action[2] = rand_action[2] * (max_lim_angle - min_lim_angle) + min_lim_angle
        
        success, traj_pos, traj_force = rope.step(rand_action)

    rope.close(str(i))
    