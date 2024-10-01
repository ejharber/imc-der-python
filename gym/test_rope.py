from rope import Rope
import matplotlib.pyplot as plt

X = [2.00049247e-01, 2.00032400e-01, .1, .5, 1e8, 0.000001, 0.05, 0.05, 0.05]

rope = Rope(X)

q0 = [180,  -54,  134, -167,  -90,    0] 
qf = [180,  -54,  134, -167,  0,    0] 

success, traj_pos_sim, traj_force_sim, traj_force_sim_base, traj_force_sim_rope, q_save, _ = rope.run_sim(q0, qf)

# plt.plot(traj_pos.T)
# plt.show()

print(q_save)
# plt.plot(q_save[:,-500:].T, 'r.')
rope.render(q_save)

plt.figure("force sim v real world data")
# plt.plot(traj_force, 'r.', label='real world')
plt.plot(traj_force_sim, 'r-', label='sim total')
plt.plot(traj_force_sim_base, 'b-', label='sim base')
plt.plot(traj_force_sim_rope, 'g-', label='sim rope')
plt.legend()

plt.show()