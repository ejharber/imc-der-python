import sys
sys.path.append("../gym/")

from rope_gym2 import RopeEnv
import numpy as np 
import multiprocessing as mp
from multiprocessing import Process

def construct_data(offset = 0):
    env = RopeEnv()

    actions_0 = []
    x_ee_0 = []
    trajectories_0 = []
    x_0 = []

    actions_1 = []
    x_ee_1 = []
    trajectories_1 = []
    x_1 = []

    actions_2 = []
    x_ee_2 = []
    trajectories_2 = []
    x_2 = []

    actions_3 = []
    x_ee_3 = []
    trajectories_3 = []
    x_3 = []

    seeds = []

    print(offset)

    for i in range(50_000):

        print(offset, i)

        env.reset(seed = i + offset)
        action_0 = np.zeros(3)
        success, trajectory_0 = env.rope.step(action_0)
        state_0 = env.rope.getState()

        if not success: continue 

        action_1 = env.action_space.sample()
        success, trajectory_1 = env.rope.step(action_1)
        state_1 = env.rope.getState()

        if not success: continue 

        action_2 = env.action_space.sample()
        success, trajectory_2 = env.rope.step(action_2)
        state_2 = env.rope.getState()

        if not success: continue 

        action_3 = env.action_space.sample()
        success, trajectory_3 = env.rope.step(action_3)
        state_3 = env.rope.getState()

        if not success: continue 

        actions_0.append(action_0)
        x_ee_0.append(state_0.x_ee)
        trajectories_0.append(trajectory_0)
        x_0.append(state_0.x)

        actions_1.append(action_1)
        x_ee_1.append(state_1.x_ee)
        trajectories_1.append(trajectory_1)
        x_1.append(state_1.x)

        actions_2.append(action_2)
        x_ee_2.append(state_2.x_ee)
        trajectories_2.append(trajectory_2)
        x_2.append(state_2.x)

        actions_3.append(action_3)
        x_ee_3.append(state_3.x_ee)
        trajectories_3.append(trajectory_3)
        x_3.append(state_3.x)

        seeds.append(i + offset)

    np.savez("rope_motion_noise_" + str(offset), actions_0=actions_0, x_ee_0=x_ee_0, trajectories_0=trajectories_0, x_0=x_0,
                                            actions_1=actions_1, x_ee_1=x_ee_1, trajectories_1=trajectories_1, x_1=x_1,
                                            actions_2=actions_2, x_ee_2=x_ee_2, trajectories_2=trajectories_2, x_2=x_2,
                                            actions_3=actions_3, x_ee_3=x_ee_3, trajectories_3=trajectories_3, x_3=x_3,
                                            seeds=seeds)

pool = mp.Pool(processes=4)

args = []

for i in range(12):
    args.append(50_000 * i)

pool.map(construct_data, args)
