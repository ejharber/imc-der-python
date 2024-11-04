import sys
sys.path.append("../gym/")
from rope import Rope

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import differential_evolution

def cost_fun(params, q0_save, qf_save, traj_robot_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save):

    for i in range(2, q0_save.shape[0]):

        q0 = q0_save[i, :]
        qf = qf_save[i, :]

        traj_robot = traj_robot_save[i, :, :]
        traj_rope_base = traj_rope_base_save[i, round(params[-1]):round(params[-1] + 500), :] # taken by mocap
        traj_rope_tip = traj_rope_tip_save[i, round(params[-1]):round(params[-1] + 500), :] # taken by mocap
        traj_force = traj_force_save[i, round(params[-2]):round(params[-2] + 500), :]

        rope = Rope(params[:-2])        
        success, traj_pos_sim, traj_force_sim, traj_force_sim_base, traj_force_sim_rope, q_save, _ = rope.run_sim(q0, qf)

        print(traj_pos_sim.shape, traj_force_sim.shape, traj_force_sim_base.shape, traj_force_sim_rope.shape)

        # print(traj_force_sim_rope)
        # rope.render(q_save, traj_rope_base, traj_pos_sim)
        rope.render(q_save, traj_rope_tip)

        plt.figure("pose sim v real world data")
        plt.plot(traj_rope_tip, 'r-', label='real world')
        plt.plot(traj_pos_sim, 'b-', label='sim')
        plt.legend()

        plt.figure("force sim v real world data")
        plt.plot(traj_force, 'r.', label='real world')
        plt.plot(traj_force_sim, 'r-', label='sim total')
        plt.plot(traj_force_sim_base, 'b-', label='sim base')
        plt.plot(traj_force_sim_rope, 'g-', label='sim rope')
        plt.legend()

        plt.show()

        # exit()

if __name__ == "__main__":

    folder_name = "filtered_data"
    file = "N2.npz"
    file_name = folder_name + "/" + file

    data = np.load(file_name)

    traj_robot_tool_save = data["traj_robot_tool_save"]
    traj_rope_base_save = data["traj_rope_base_save"]
    traj_rope_tip_save = data["traj_rope_tip_save"]    
    traj_force_save = data["traj_force_save"]
    q0_save = data["q0_save"]
    qf_save = data["qf_save"]

    params = np.load("params/N2.npz")["params"]
    print(params)
    # exit()
    # params = [.254, .2413, 0.003, 0.03]

    # params = [inertial_params[0], inertial_params[1], 0.5842, 0.01, 1e4, 0.01, inertial_params[3], .0103, .06880, inertial_params[4], 60, 60]
    # params = [0.22915540232398574, 0.20000000000000004, 0.036294360943663645, 0.7037556574462394, 58032633.295201994, 0.47778348511389546, 0.05314106679722293, 0.015495622626538724, 0.01684803919905322, 46.16667651912822, 45.76536688106134]
    # params = [0.19315186293715925, 0.15165394764203527, 0.5914301717039617, 0.21938201785236222, 9203344.337211214, 0.214039657881742, 0.054183350275864156, 0.010647563423367076, 0.0509047146544107, 53.37315012304928, 43.99597956691203]
    # params = [1.35495433e-04, 1.38784756e-01, 5.79650728e-01, 4.84518153e-03,
       # 1.10808512e+02, 2.41103906e+04, 2.21659738e+04, 1.39608865e-03,
       # 4.25843588e+01, 2.47631809e-01, 9.13829708e-02, 3.07163269e-04,
       # 5.88072600e-02, 4.34659659e+01, 5.20962321e+01]
    # self.dL_stick = X[0]
    # self.dL_ati = X[1] # should replace this with an array         
    # self.dL_rope = X[2]
    # self.Kb = X[3]
    # self.Ks = X[4]
    # self.damp = X[5]
    # self.m_holder = X[6]
    # self.m_rope = X[7]
    # self.m_tip = X[8]

    cost_fun(params, q0_save, qf_save, traj_robot_tool_save, traj_rope_base_save, traj_rope_tip_save, traj_force_save)