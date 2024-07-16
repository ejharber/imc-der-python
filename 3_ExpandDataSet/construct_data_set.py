import sys
sys.path.append("../gym/")
from rope import Rope

import numpy as np 
import multiprocessing as mp
from multiprocessing import Process

def construct_data(offset = 0):

    params = np.load("../2_SysID/res_all_noise.npz")["x"]
    rope = Rope(params)

    seeds = []

    q0_save = []
    qf_save = []
    traj_pos_save = []
    traj_force_save = []
    qs_save = []
    fs_save = []

    for i in range(10_000):

        print(offset, i)

        np.random.seed(i + offset)

        q0 = [180, -53.25, 134.66, -171.28, -90, 0]
        qf = [180, -90, 100, -180, -90, 0]
        qf[1] = qf[1] + np.random.rand() * 20 - 10
        qf[2] = qf[2] + np.random.rand() * 20 - 10
        qf[3] = qf[3] + np.random.rand() * 24 - 12

        success, traj_pos, traj_force, qs, fs = rope.run_sim(q0, qf)

        if not success: continue 
        seeds.append(i + offset)

        q0_save.append(q0)
        qf_save.append(qf)
        traj_pos_save.append(traj_pos)
        traj_force_save.append(traj_force)
        qs_save.append(qs)
        fs_save.append(fs)

    np.savez("expanded_data/rope_motion_" + str(offset), q0_save=q0_save, qf_save=qf_save, traj_pos_save=traj_pos_save, traj_force_save=traj_force_save,
                                                qs_save=qs_save, fs_save=fs_save, seeds=seeds)

pool = mp.Pool(processes = 24)

args = []

for i in range(12*5):
    args.append(10_000 * i)

pool.map(construct_data, args)

# construct_data()
