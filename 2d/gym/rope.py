import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
from scipy.interpolate import PchipInterpolator, CubicSpline
import imageio
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
            
            # enf effector states
            self.x_1_ee = self.x_1[-1]
            self.x_2_ee = self.x_2[-1]
            self.x_ee = np.array([self.x_1_ee, self.x_2_ee])

            # robot states
            self.x_1_robot = self.x[0]
            self.x_2_robot = self.x[1]
            q_ = self.q[:2] - self.q[2:4]
            self.x_theta_robot = np.arctan2(q_[1], q_[0]) 
            if self.x_theta_robot >= 2*np.pi:
                self.x_theta_robot -= 2*np.pi
            if self.x_theta_robot < 0:
                self.x_theta_robot += 2*np.pi
            self.x_robot = [self.x_1_robot, self.x_2_robot, self.x_theta_robot]

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
        self.traj_pos = np.zeros((20, 101))
        self.traj_force = np.zeros((2, 101))

        self.save_render = []

    def setState(self, state):

        self._simulation = state 

        self.render()

    def getState(self):
        
        return self._simulation

    def step(self, weighpoints):

        state = self.getState()

        traj_u = []

        time = .5 # set sim time to be 0.5 seconds

        for dim in range(weighpoints.shape[0]):
            x_1_traj = [0, time]
            x_2_traj = [state.x_robot[dim], weighpoints[dim]]

            print(dim, x_2_traj)
            f_prime = CubicSpline(x_1_traj, x_2_traj, bc_type='clamped').derivative(1)
            traj_u.append(f_prime(np.linspace(0, time, round(time/self.dt) + 1)))

        traj_u = np.array(traj_u)
        f_save, q_save, u_save, success = run_simulation(state.x, state.u, self.N, self.dt, self.length, self.dL, self.R, self.g, self.EI, self.EA, self.damp, self.m, traj_u = traj_u)

        if not success:
            print("FAILED")
            return success, [], []

        self.traj_pos = q_save
        force_x = np.sum(f_save[5::2, :], axis = 0)
        force_y = np.sum(f_save[4::2, :], axis = 0)
        self.traj_force = np.array([force_x, force_y])

        if self.render_mode is not None and (self.render_mode == "Human" or self.render_mode == "Both"):
            for i in range(0, q_save.shape[1], 10):
                x = q_save[:, i]
                u = u_save[:, i]
                state = self.State(x, u)
                self.setState(state)

        x = q_save[:, -1]
        u = u_save[:, -1]

        state = self.State(x, u)
        self.setState(state)

        print("done", state.x_theta_robot)

        return success, self.traj_pos, self.traj_force

    def reset(self, random_sim_params, seed = None):

        np.random.seed(seed)

        if random_sim_params:
            self.dL = 0.0046*2.1*np.random.uniform(.95, 1.05) # should get these from sim
            self.length = (self. N - 1) * self.dL
            self.EI = 1e-2*np.random.uniform(.95, 1.05) # should replace this with an array LATER
            self.EA = 1e7*np.random.uniform(.95, 1.05) # should replace this with an array LATER
            self.damp = 0.15*np.random.uniform(.95, 1.05)
            self.m = 0.2*np.random.uniform(.95, 1.05) # should replace this with an array LATER

        x = np.zeros((self.N, 2))
        for c in range(self.N):
            x[c, 1] = - c * self.dL
        x = x.flatten()

        u = np.zeros(self.N*2)

        state = self.State(x, u)
        self.setState(state)

        render_mode = self.render_mode
        self.render_mode = None # turn render mode off for reset
        for _ in range(5):
            self.step(np.array([0,0,0])) # reach a steady state before beginning        # helps with force peaks
        self.render_mode = render_mode

        self.render()

        return self.getState()

    def render(self, test = None):
        if self.render_mode is None:
            return

        state = self.getState()

        if self.render_mode == "Human":
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
                plt.draw()
                plt.pause(0.001) 
                self.fig.canvas.draw()


        elif self.render_mode == "Computer":
            if self.fig is None:
                sns.set() # Setting seaborn as default style even if use only matplotlib
                self.fig, self.ax = plt.subplots(2, 2, figsize=(5, 5))

                self.ax[0, 0].pcolormesh(self.traj_pos[0::2], edgecolors='w', linewidth=.2)
                self.ax[0, 0].set_title("Pose Trajectory (x)")
                self.ax[0, 1].pcolormesh(self.traj_pos[1::2], edgecolors='w', linewidth=.2)
                self.ax[0, 1].set_title("Pose Trajectory (y)")
                self.ax[1, 0].pcolormesh(self.traj_force[0::2], edgecolors='w', linewidth=.2)
                self.ax[1, 0].set_title("Force Trajectory (x)")
                self.ax[1, 1].pcolormesh(self.traj_force[1::2], edgecolors='w', linewidth=.2)
                self.ax[1, 1].set_title("Force Trajectory (y)")

                plt.show(block=False)

                plt.draw()
                plt.pause(0.001)

            else:            
                self.ax[0, 0].pcolormesh(self.traj_pos[0::2], edgecolors='w', linewidth=.2)
                self.ax[0, 1].pcolormesh(self.traj_pos[1::2], edgecolors='w', linewidth=.2)
                self.ax[1, 0].pcolormesh(self.traj_force[0::2], edgecolors='w', linewidth=.2)
                self.ax[1, 1].pcolormesh(self.traj_force[1::2], edgecolors='w', linewidth=.2)

                plt.draw()
                plt.pause(0.001) 
                self.fig.canvas.draw()

        elif self.render_mode == "Both":
            if self.fig is None:
                sns.set() # Setting seaborn as default style even if use only matplotlib
                self.fig, self.ax = plt.subplot_mosaic("AABC;AADE", figsize=(10, 5))

                self.circles = []
                self.circles += [patches.Circle((state.x_1[0], state.x_2[0]), radius=self.radius, ec = None, fc='red')]
                self.ax["A"].add_patch(self.circles[-1])
                for n in range(1, self.N-1):
                    self.circles += [patches.Circle((state.x_1[n], state.x_2[n]), radius=self.radius, ec = None, fc='blue')]
                    self.ax["A"].add_patch(self.circles[-1])
                self.circles += [patches.Circle((state.x_1[n], state.x_2[n]), radius=self.radius, ec = None, fc='green')]
                self.ax["A"].add_patch(self.circles[-1])

                lim = .4
                self.ax["A"].axis('equal')
                self.ax["A"].set_xlim(-lim, lim)
                self.ax["A"].set_ylim(-lim, lim)

                self.ax["B"].pcolormesh(self.traj_pos[0::2], edgecolors='w', linewidth=.2)
                self.ax["B"].set_title("Pose Trajectory (x)")
                self.ax["C"].pcolormesh(self.traj_pos[1::2], edgecolors='w', linewidth=.2)
                self.ax["C"].set_title("Pose Trajectory (y)")
                self.ax["D"].pcolormesh(self.traj_force[0::2], edgecolors='w', linewidth=.2)
                self.ax["D"].set_title("Force Trajectory (x)")
                self.ax["E"].pcolormesh(self.traj_force[1::2], edgecolors='w', linewidth=.2)
                self.ax["E"].set_title("Force Trajectory (y)")

                plt.show(block=False)
                plt.tight_layout()

                plt.draw()
                plt.pause(0.001)

            else:            
                for n, circle in enumerate(self.circles):
                    circle.set(center=(state.x_1[n], state.x_2[n]))

                self.ax["B"].pcolormesh(self.traj_pos[0::2], edgecolors='w', linewidth=.2)
                self.ax["C"].pcolormesh(self.traj_pos[1::2], edgecolors='w', linewidth=.2)
                self.ax["D"].pcolormesh(self.traj_force[0::2], edgecolors='w', linewidth=.2)
                self.ax["E"].pcolormesh(self.traj_force[1::2], edgecolors='w', linewidth=.2)

                plt.draw()
                plt.pause(0.001) 
                self.fig.canvas.draw()

        else:
            print("unknown render_mode")

        # Later
        # self.fig.canvas.draw()
        # image = np.array(self.fig.canvas.renderer.buffer_rgba())
        # if image.shape[0] == 750:
        #     self.save_render.append(image)
        # print(image.shape)
        # # now you have a numpy array representing the rendered image
        # # print(image.shape)  # (height, width, channels)

    def close(self):
        if self.render_mode is not None:
            plt.close(self.fig)
            imageio.mimsave('animation.gif', self.save_render, duration=self.dt*1000)

