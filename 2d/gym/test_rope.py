from rope import RopePython
import numpy as np
import matplotlib.pyplot as plt
import time

rope = RopePython("Both")
rope.reset(random_sim_params = False)

for _ in range(100):
    success, traj_pos, traj_force = rope.step(np.array([0, 0, 0]))
    time.sleep(1)
print(traj_pos.shape)
print(traj_force.shape)

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].imshow(traj_pos[0::2])
# axs[0, 1].imshow(traj_pos[1::2])
# axs[1, 0].imshow(traj_force[0::2])
# axs[1, 1].imshow(traj_force[1::2])
# plt.show()
# plt.plot(traj_force.T)
# plt.show()

# n_cells = 800
# lim = 0.4

# q_x = q[0::2, :]
# q_x = np.round((q_x + lim) * 800)
# q_x = np.array(q_x.flatten(), dtype=int)

# q_y = q[1::2, :]
# q_y = np.round((q_y + lim) * 800)
# q_y = np.array(q_y.flatten(), dtype=int)

# occupancy_grid = np.zeros((n_cells, n_cells), dtype=int)

# occupancy_grid[q_y, -q_x] = 1


# plt.subplots(figsize=(5, 5))
# plt.imshow(occupancy_grid, cmap='binary')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Occupancy Grid')
# plt.show()
