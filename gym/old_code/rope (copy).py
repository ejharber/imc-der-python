import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
from scipy.interpolate import PchipInterpolator, CubicSpline, splrep
import imageio
import seaborn as sns

from numpy_sim import *

class Rope(object):
    def __init__(self, random_sim_params, render_mode = None):

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

    def step(self, weighpoints):

        f_save, q_save, u_save, success = run_simulation(state.x, state.u, self.N, self.dt, self.length, self.dL, self.R, self.g, self.EI, self.EA, self.damp, self.m, traj_u = traj_u)

        force_x = np.sum(f_save[5::2, :], axis = 0)
        force_y = np.sum(f_save[4::2, :], axis = 0)

        x = q_save[:, -1]
        u = u_save[:, -1]

        return success, traj_pos, traj_force

    def reset(self, seed = None):

        x = np.zeros((self.N, 2))
        for c in range(self.N):
            x[c, 1] = - c * self.dL
        x = x.flatten()
        u = np.zeros(self.N*2)

        state = self.State(x, u)
        self.setState(state)

        for _ in range(3):
            self.step(np.array([0,0,0])) # reach a steady state before beginning        # helps with force peaks

        self.render_mode = render_mode

        return self.getState()
