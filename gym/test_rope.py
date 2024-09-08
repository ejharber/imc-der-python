from rope import Rope
import matplotlib.pyplot as plt

X = [5.00005e-02,5.00005e-02,5.00005e-02,5.00000e+03,5.00000e+03,5.00000e+06,5.00000e+06,5.00000e+00,5.00005e-02,5.00005e-02,5.00005e-02]
rope = Rope(X)

q0 = [0,  -54,  134, -167,  -90,    0] 
qf = [0,  -54,  134, -167,  0,    0] 

success, traj_pos, traj_force, q_save, f_save = rope.run_sim(q0, qf)

# plt.plot(traj_pos.T)
# plt.show()

# print(q_save.shape)
# plt.plot(q_save[:,-500:].T, 'r.')
# rope.render(q_save)

# print(traj_force)
# plt.plot(traj_force)
# plt.show()