from rope import RopePython
import numpy as np
import matplotlib.pyplot as plt
import time

max_lim = 0.2
min_lim = - max_lim

max_lim_angle = 3 / 2 * np.pi
min_lim_angle = - max_lim_angle

for i in range(100):
    rope = RopePython(True)
    rope.reset(seed = i)
    np.random.seed(i)
    rand_action = np.random.random((3, 1))
    rand_action[:2] = rand_action[:2] * (max_lim - min_lim) - max_lim
    rand_action[2] = max_lim_angle * np.random.choice([-1, 1])
    
    print(i, rand_action)
    # rand_action += np.random.random(rand_action.shape) * 1e-2
    # print(rand_action)
    success, traj_pos, traj_force = rope.step(rand_action)

    if not success:
        np.save(str(i), rand_action)
        break
    # print(rand_action, success)
    # time.sleep(1)
# print(traj_pos.shape)
# print(traj_force.shape)

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
