from rope import Rope
import matplotlib.pyplot as plt

X = [2.00049247e-01, 2.00032400e-01, .1, 1e-1, 1e8, 0.000001, 0.05, 0.05, 0.05]

rope = Rope(X)

q0 = [0,  -54,  134, -167,  -90,    0] 
qf = [0,  -54,  134, -167,  0,    0] 

success, traj_pos, traj_force, q_save, f_save = rope.run_sim(q0, qf)

# plt.plot(traj_pos.T)
# plt.show()

# print(q_save.shape)
# plt.plot(q_save[:,-500:].T, 'r.')
rope.render(q_save)

# # print(traj_force)
# # plt.plot(traj_force)
# plt.show()