from mass_spring import MassSpring
import matplotlib.pyplot as plt

X = [0.5, 10, 0, 0.2, 0.5]

massspring = MassSpring(X)

success, traj_pos_sim, traj_force_sim, q_save, _ = massspring.run_sim()

# plt.plot(traj_pos.T)
# plt.show()

print(q_save)
# plt.plot(q_save[:,-500:].T, 'r.')
massspring.render(q_save)

plt.figure("force sim v real world data")
# plt.plot(traj_force, 'r.', label='real world')
plt.plot(traj_pos_sim, 'r-', label='sim total')
plt.legend()

print(traj_force_sim)
plt.figure("force sim v real world data")
# # plt.plot(traj_force, 'r.', label='real world')
plt.plot(traj_force_sim, 'r-', label='sim total')
# plt.plot(traj_force_sim_base, 'b-', label='sim base')
# plt.plot(traj_force_sim_rope, 'g-', label='sim rope')
# plt.legend()

plt.show()