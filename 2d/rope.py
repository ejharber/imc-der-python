import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
from scipy.interpolate import PchipInterpolator, CubicSpline

import seaborn as sns

from numpy_sim import *
# from cupy_sim import *

class RopePython(object):

    class State:
        """
        Rope full state space
        """
        def __init__(self, x, u):
            
            # states for loading a simulation 
            self.x = x
            self.u = u
            self.q = np.concatenate((self.x, self.u))

            # states for analysis
            self.x_1 = self.x[0::2]
            self.x_2 = self.x[1::2]
            self.x_1_ee = self.x_1[-1]
            self.x_2_ee = self.x_2[-1]
            # print(self.x_1_ee.shape)
            self.x_ee = np.array([self.x_1_ee, self.x_2_ee])
            if len(self.x_ee.shape) > 1:
                self.x_ee = self.x_ee[:, 0]
            # print(self.x_ee.shape)

    def __init__(self, render_mode = None):

        ## Model Parameters (Calculated)
        self.N = 10
        self.dt = 0.005
        self.R = 0.0046
        self.radius = self.R
        self.g = 9.8

        self.dL = 0.0046*2.1 # should replace this with an array         
        self.length = (self. N - 1) * self.dL
        self.EI = 1e-2 # should replace this with an array LATER
        self.EA = 1e7 # should replace this with an array LATER
        self.damp = 0.15
        self.m = 0.2 # should replace this with an array LATER

        self._simulation = None

        self.render_mode = render_mode
        self.fig, self.ax = None, None

        self.save_render = []

        self.reset()

    def setState(self, state):

        self._simulation = state 

        self.render()

    def getState(self):
        
        return self._simulation

    def step(self, weighpoints):

        state = self.getState()

        weighpoints[2] = (weighpoints[2] + np.pi) % (2*np.pi) - np.pi

        traj_velocity = []

        time = .5

        for dim in range(weighpoints.shape[0]):
            x_traj = [0, time]
            y_traj = [0, weighpoints[dim]]
            f_prime = CubicSpline(x_traj, y_traj, bc_type='clamped').derivative(1)
            traj_velocity.append(f_prime(np.linspace(0, time, round(time/0.01) + 1)))

        traj_velocity = np.array(traj_velocity)

        traj_u = traj_velocity
        f_save, q_save, u_save, success = run_simulation(state.x, state.u, self.N, self.dt, self.length, self.dL, self.R, self.g, self.EI, self.EA, self.damp, self.m, traj_u = traj_u)

        if not success:
            print("FAILED")
            return success 

        force_y = np.sum(f_save[4::2, :], axis = 0)
        force_x = np.sum(f_save[5::2, :], axis = 0)

        force = force_x * force_x + force_y * force_y
        force = np.power(force, 0.5)

        if self.render is not None:

            for i in range(q_save.shape[1]):
                x = q_save[:, i]
                u = u_save[:, i]
                state = self.State(x, u)
                self.setState(state)

        else:
            x = q_save[:, -1]
            u = u_save[:, -1]

            state = self.State(x, u)
            self.setState(state)

        return force, q_save, u_save, success
        return success

    def reset(self, seed = None):

        np.random.seed(seed)
        self.dL = 0.0046*np.random.uniform(2.1, 4) # should get these from sim

        x = np.zeros((self.N, 2))
        for c in range(self.N):
            x[c, 1] = - c * self.dL
        x = x.flatten()

        u = np.zeros(self.N*2)

        state = self.State(x, u)
        self.setState(state)

        return self.getState()

    def render(self, test = None):
        if self.render_mode is None:
            return

        state = self.getState()

        if self.render_mode == "Human":
            plt.figure("Human")
            if self.fig is None:
                sns.set() # Setting seaborn as default style even if use only matplotlib
                self.fig, self.ax = plt.subplots(figsize=(5, 5))

                self.circles = []
                self.circles += [patches.Circle((state.x_1[0], state.x_2[0]), radius=self.radius, ec = None, fc='red')]
                self.ax.add_patch(self.circles[-1])
                for n in range(1, self.N-1):
                    self.circles += [patches.Circle((state.x_1[n], state.x_2[n]), radius=self.radius, ec = None, fc='blue')]
                    self.ax.add_patch(self.circles[-1])
                self.circles += [patches.Circle((state.x_1[n], state.x_2[n]), radius=self.radius, ec = None, fc='green')]
                self.ax.add_patch(self.circles[-1])
                # self.goal_circle = patches.Circle((self.goal[0], self.goal[1]), radius=self.radius, ec = 'black', fc=None)
                # self.ax.add_patch(self.goal_circle)
                # self.goal_circle = self.ax.plot(self.goal[0], self.goal[1], 'g', marker = '+', markersize=10, markeredgewidth=2)[0]

                lim = .4
                self.ax.axis('equal')
                self.ax.set_xlim(-lim, lim)
                self.ax.set_ylim(-lim, lim)
                plt.show(block=False)

                plt.draw()
                plt.pause(0.001)

            else:
                
                for n, circle in enumerate(self.circles):
                    circle.set(center=(state.x_1[n], state.x_2[n]))
                # self.goal_circle.set_xdata(self.goal[0])     
                # self.goal_circle.set_ydata(self.goal[1])     
                plt.draw()
                plt.pause(0.001) 
                self.fig.canvas.draw()


        # if render == "Occupancy":


        # Later
        # self.fig.canvas.draw()
        # image = np.array(self.fig.canvas.renderer.buffer_rgba())
        # if image.shape[0] == 750:
        #     self.save_render.append(image)
        # # print(image.shape)
        # # now you have a numpy array representing the rendered image
        # # print(image.shape)  # (height, width, channels)

    def close(self):
        if self.render:
            plt.close(self.fig)
            imageio.mimsave('animation.gif', self.save_render, duration=self.dt*1000)

