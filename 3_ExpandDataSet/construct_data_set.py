import sys
sys.path.append("../gym/")
from rope import Rope

import numpy as np 
import multiprocessing as mp
from multiprocessing import Process

def construct_data(offset = 0):

    params = np.load("../2_SysID/res_all.npz")["x"]
    UR5e = UR5eCustom()
    rope = Rope(params)

    actions_0 = []
    traj_pos_0 = []
    traj_force_0 = []
    
    actions_1 = []
    traj_pos_1 = []
    traj_force_1 = []

    actions_2 = []
    traj_pos_2 = []
    traj_force_2 = []

    actions_3 = []
    traj_pos_3 = []
    traj_force_3 = []

    actions_4 = []
    traj_pos_4 = []
    traj_force_4 = []

    actions_5 = []
    traj_pos_5 = []
    traj_force_5 = []

    actions_6 = []
    traj_pos_6 = []
    traj_force_6 = []

    actions_7 = []
    traj_pos_7 = []
    traj_force_7 = []

    actions_8 = []
    traj_pos_8 = []
    traj_force_8 = []

    actions_9 = []
    traj_pos_9 = []
    traj_force_9 = []

    actions_10 = []
    traj_pos_10 = []
    traj_force_10 = []

    actions_11 = []
    traj_pos_11 = []
    traj_force_11 = []

    seeds = []

    print(offset)

    for i in range(10_000):

        print(offset, i)

        np.seed(i + offset)
        seeds.append(i + offset)

        q0 = data["q0_save"]
        qf = [180, -90, 100, -180, -90, 0]
        qf[1] = qf[1] + np.random.rand() * 20 - 10
        qf[2] = qf[2] + np.random.rand() * 20 - 10
        qf[3] = qf[3] + np.random.rand() * 24 - 12
        success, traj_pos_sim, traj_force_sim, q_save, _ = rope.run_sim(q0, qf)

        observation_state_0 = env.reset(i + offset)
        if np.all(observation_state_0["pos_traj"] == 0): continue 

        actions_0.append(observation_state_0["action"])
        traj_pos_0.append(observation_state_0["pos_traj"])
        traj_force_0.append(observation_state_0["force_traj"])

    np.savez("data/rope_motion_noise_" + str(offset), actions_0=actions_0, traj_pos_0=traj_pos_0, traj_force_0=traj_force_0,
                                                      actions_1=actions_1, traj_pos_1=traj_pos_1, traj_force_1=traj_force_1,
                                                      actions_2=actions_2, traj_pos_2=traj_pos_2, traj_force_2=traj_force_2,
                                                      actions_3=actions_3, traj_pos_3=traj_pos_3, traj_force_3=traj_force_3,
                                                      actions_4=actions_4, traj_pos_4=traj_pos_4, traj_force_4=traj_force_4,
                                                      actions_5=actions_5, traj_pos_5=traj_pos_5, traj_force_5=traj_force_5,
                                                      actions_6=actions_6, traj_pos_6=traj_pos_6, traj_force_6=traj_force_6,
                                                      actions_7=actions_7, traj_pos_7=traj_pos_7, traj_force_7=traj_force_7,
                                                      actions_8=actions_8, traj_pos_8=traj_pos_8, traj_force_8=traj_force_8,
                                                      actions_9=actions_9, traj_pos_9=traj_pos_9, traj_force_9=traj_force_9,
                                                      actions_10=actions_10, traj_pos_10=traj_pos_10, traj_force_10=traj_force_10,
                                                      actions_11=actions_11, traj_pos_11=traj_pos_11, traj_force_11=traj_force_11,
                                                      seeds=seeds)

# pool = mp.Pool(processes = 1)

# args = []

# for i in range(12*5):
    # args.append(10_000 * i)

# pool.map(construct_data, args)

construct_data()
