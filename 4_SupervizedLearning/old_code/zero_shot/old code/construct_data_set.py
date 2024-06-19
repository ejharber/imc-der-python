import sys
sys.path.append("../../gym/")

from rope_gym import RopeEnv
from rope import RopePython
import numpy as np 
import multiprocessing as mp
from multiprocessing import Process

def construct_data(offset = 0):

    actions = []
    trajs_pos = []
    trajs_force = []

    seeds = []

    print(offset)

    for i in range(10_000):

        print(offset, i)

        env = RopeEnv(True)
        env.reset(seed = i + offset)
        action = env.action_space.sample()
        observation_state, reward, terminate, _ = env.step(action)

        actions.append(action)
        trajs_pos.append(observation_state["pos_traj"])
        trajs_force.append(observation_state["force_traj"])

        print(np.array(actions).shape, np.array(trajs_pos).shape, np.array(trajs_force).shape)

        seeds.append(i + offset)

    # print(np.array(actions).shape, np.array(trajs_pos).shape, np.array(trajs_force).shape)

    actions = np.array(actions)
    trajs_pos = np.array(trajs_pos)
    trajs_force = np.array(trajs_force)

    print(actions.shape, trajs_pos.shape, trajs_force.shape)

    np.savez("data/rope_motion_" + str(offset), actions=actions, trajs_pos=trajs_pos, trajs_force=trajs_force, seeds=seeds)

# pool = mp.Pool(processes = 1)

# args = []

for i in range(12*5):
    # args.append(10_000 * i)
    construct_data(10_000 * i)
# pool.map(construct_data, args)
