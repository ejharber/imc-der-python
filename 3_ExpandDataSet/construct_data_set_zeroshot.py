import sys
sys.path.append("../gym/")
from rope import Rope

import os
import numpy as np 
import multiprocessing as mp
from multiprocessing import Pool

def construct_data(offset, dataset_name, save_file, data_set_size):
    # Load simulation parameters
    params = np.load(f"../2_SysID/params/{dataset_name}.npz")["params"]
    rope = Rope(params)

    # Data storage lists
    seeds = []
    q0_save = []
    qf_save = []
    traj_pos_save = []
    traj_force_save = []
    traj_force_base_save = []
    traj_force_rope_save = []
    qs_save = []
    fs_save = []

    # Run simulations
    for i in range(data_set_size):
        print(f"Dataset: {dataset_name}, Offset: {offset}, Iteration: {i}/{data_set_size}")
        np.random.seed(i + offset)

        # Define initial and randomized target configurations
        q0 = [180, -80.55, 138.72, -148.02, -90, 0]
        qf = [180.0, -100.0, 100.0, -180.0, -90.0, 0.0]
        qf[1] += np.random.uniform(-12, 12)
        qf[2] += np.random.uniform(-12, 12)
        qf[3] += np.random.uniform(-12, 12)
        # qf[1] += np.random.uniform(-8, 8)
        # qf[2] += np.random.uniform(-8, 8)
        # qf[3] += np.random.uniform(-8, 8)
        # qf[1] += np.random.choice([-1, 1]) * np.random.uniform(8, 10)
        # qf[2] += np.random.choice([-1, 1]) * np.random.uniform(8, 10)
        # qf[3] += np.random.choice([-1, 1]) * np.random.uniform(8, 10)

        # Run simulation
        success, traj_pos_sim, traj_force_sim, traj_force_sim_base, traj_force_sim_rope, q_save, f_save = rope.run_sim(q0, qf)

        if not success:
            continue

        # Append data to lists
        seeds.append(i + offset)
        q0_save.append(q0)
        qf_save.append(qf)
        traj_pos_save.append(traj_pos_sim)
        traj_force_save.append(traj_force_sim)
        traj_force_base_save.append(traj_force_sim_base)
        traj_force_rope_save.append(traj_force_sim_rope)
        qs_save.append(q_save)
        fs_save.append(f_save)

    # Save all data into a .npz file
    np.savez(f"{save_file}/{offset}",
             q0_save=q0_save, qf_save=qf_save,
             traj_pos_save=traj_pos_save, traj_force_save=traj_force_save,
             traj_force_base_save=traj_force_base_save, traj_force_rope_save=traj_force_rope_save,
             qs_save=qs_save, fs_save=fs_save, seeds=seeds)

if __name__ == "__main__":

    dataset_name = "N2_pose_80"
    save_file = dataset_name + "_zs"
    data_set_size = 100

    # Ensure directory exists
    os.makedirs(save_file, exist_ok=True)

    pool = Pool(processes=10)
    args = [data_set_size * i for i in range(1000)]
    pool.starmap(construct_data, [(arg, dataset_name, save_file, data_set_size) for arg in args])