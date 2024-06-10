from rope import Rope
import matplotlib.pyplot as plt

X = [0.05, 1, 1e-2, 1e7, 1e7, 0.15, 0.2, 0.2, 0.2]

# X = [0.005, 1e-2, 1e7, 0.15, 0.2, 0.2, 0.2]
rope = Rope(X)

q0 = [0,  -54,  134, -167,  -90,    0] 
qf = [0,  -54,  134, -167,  0,    0] 

# qf = [0,  -88.33333333,   93.33333333, -183.33333333,  -90,    0]


success, traj_pos, traj_force, q_save, f_save = rope.run_sim(q0, qf)

# plt.plot(traj_pos.T)
# plt.show()

# print(q_save.shape)
# plt.plot(q_save[:,-500:].T, 'r.')
rope.render(q_save)

print(traj_force)
plt.plot(traj_force)
plt.show()