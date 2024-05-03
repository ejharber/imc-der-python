import numpy as np
from numpy_sim import *

import sys
sys.path.append("../UR5e")
from CustomRobots import *

class Rope(object):
    def __init__(self, random_sim_params=False):

        self.UR5e = UR5eCustom()

        self.random_sim_params = random_sim_params
        ## Model Parameters (Calculated)
        self.N = 10
        self.dt = 0.001 # we are collecting data at 500 hz
        self.g = 9.8

        ## Physics Parameters
        self.dL = 0.005 # should replace this with an array         
        self.length = (self. N - 1) * self.dL
        self.EI = 1e-2
        self.EA = 1e7
        self.damp = 0.15
        self.m1 = 0.2
        self.m2 = 0.2
        self.m3 = 0.2

        self.x0 = None
        self.u0 = None

        self.reset()

    def run_sim(self, q0, qf):

        traj = self.UR5e.create_trajectory(q0, qf)
        traj = self.UR5e.fk_traj_stick(traj, True)

        f_save, q_save, u_save, success = run_simulation(self.x0, self.u0, self.N, self.dt, self.length, self.dL, self.g, self.EI, self.EA, self.damp, self.m, traj_u = traj_u)

        force_x = np.sum(f_save[5::2, :], axis = 0)
        force_y = np.sum(f_save[4::2, :], axis = 0)

        x = q_save[:, -1]
        u = u_save[:, -1]

        return success, traj_pos, traj_force

    def reset(self, seed = None):

        self.x0 = np.zeros((self.N, 2))
        for c in range(self.N):
            self.x0[c, 1] = - c * self.dL
        self.x0 = self.x0.flatten()
        self.u0 = np.zeros(self.N*2)

