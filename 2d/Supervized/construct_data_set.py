import sys
sys.path.append("../../gym/")

from rope_gym import RopeEnv
import numpy as np 
import multiprocessing as mp
from multiprocessing import Process

def construct_data(offset = 0):
    env = RopeEnv(True)

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

    seeds = []

    print(offset)

    for i in range(10_000):

        print(offset, i)

        observation_state_0 = env.reset(seed = i + offset)
        if np.all(observation_state_0["pos_traj"] == 0): continue 

        action = env.action_space.sample() # sample original action

        observation_state_1, _, _, _ = env.step(action)
        if np.all(observation_state_1["pos_traj"] == 0): continue 

        observation_state_2, _, _, _ = env.step(action + env.action_space.sample())
        if np.all(observation_state_2["pos_traj"] == 0): continue 

        observation_state_3, _, _, _ = env.step(action + env.action_space.sample() / 2)
        if np.all(observation_state_3["pos_traj"] == 0): continue 

        observation_state_4, _, _, _ = env.step(action + env.action_space.sample() / 10)
        if np.all(observation_state_4["pos_traj"] == 0): continue 

        observation_state_5, _, _, _ = env.step(action + env.action_space.sample() / 50)
        if np.all(observation_state_5["pos_traj"] == 0): continue 

        observation_state_6, _, _, _ = env.step(action + env.action_space.sample() / 100)
        if np.all(observation_state_6["pos_traj"] == 0): continue

        observation_state_7, _, _, _ = env.step(action + env.action_space.sample() / 200)
        if np.all(observation_state_7["pos_traj"] == 0): continue  

        observation_state_8, _, _, _ = env.step(action + env.action_space.sample() / 500)
        if np.all(observation_state_8["pos_traj"] == 0): continue  

        observation_state_9, _, _, _ = env.step(action + env.action_space.sample() / 1000)
        if np.all(observation_state_9["pos_traj"] == 0): continue  

        actions_0.append(observation_state_0["action"])
        traj_pos_0.append(observation_state_0["pos_traj"])
        traj_force_0.append(observation_state_0["force_traj"])

        actions_1.append(observation_state_1["action"])
        traj_pos_1.append(observation_state_1["pos_traj"])
        traj_force_1.append(observation_state_1["force_traj"])

        actions_2.append(observation_state_2["action"])
        traj_pos_2.append(observation_state_2["pos_traj"])
        traj_force_2.append(observation_state_2["force_traj"])

        actions_3.append(observation_state_3["action"])
        traj_pos_3.append(observation_state_3["pos_traj"])
        traj_force_3.append(observation_state_3["force_traj"])

        actions_4.append(observation_state_4["action"])
        traj_pos_4.append(observation_state_4["pos_traj"])
        traj_force_4.append(observation_state_4["force_traj"])

        actions_5.append(observation_state_5["action"])
        traj_pos_5.append(observation_state_5["pos_traj"])
        traj_force_5.append(observation_state_5["force_traj"])

        actions_6.append(observation_state_6["action"])
        traj_pos_6.append(observation_state_6["pos_traj"])
        traj_force_6.append(observation_state_6["force_traj"])

        actions_7.append(observation_state_7["action"])
        traj_pos_7.append(observation_state_7["pos_traj"])
        traj_force_7.append(observation_state_7["force_traj"])

        actions_8.append(observation_state_8["action"])
        traj_pos_8.append(observation_state_8["pos_traj"])
        traj_force_8.append(observation_state_8["force_traj"])

        actions_9.append(observation_state_9["action"])
        traj_pos_9.append(observation_state_9["pos_traj"])
        traj_force_9.append(observation_state_9["force_traj"])

        seeds.append(i + offset)

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
                                                      seeds=seeds)

pool = mp.Pool(processes = 1)

args = []

for i in range(12*5):
    args.append(10_000 * i)

pool.map(construct_data, args)
