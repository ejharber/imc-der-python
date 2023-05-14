import numpy as np
import matplotlib.pyplot as plt

from numpy_sim import *

class RopePython(object):

    class State:
        """
        Rope full state space
        """
        def __init__(self, x, u):
            
            # states for loading a simulation 
            self.x = self.x
            self.u = self.u
            self.q = np.concatenate((self.x, self.u))

            # states for analysis
            self.x_1 = self.x[0::2]
            self.x_2 = self.x[1::2]
            self.x_1_ee = self.x_1[-1]
            self.x_2_ee = self.x_2[-1]
            self.x_ee = np.array([self.x_1_ee, self.x_2_ee])

    def __init__(self, render_mode):

        ## Model Parameters (Calculated)
        self.N = 10
        self.dt = 0.01
        self.dL = 0.0046*2.1 # should get these from sim
        self.R = 0.0046 # should get these from sim
        self.length = (self. N - 1) * self.dL
        self.g = -9.8
        self.EI
        self.EA
        self.damp
        self.m

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

    def step(self, next_action):



        return self._simulation

    def reset(self, seed: Optional[int] = None):

        ## Geometry
        x = np.zeros((self.N, 2))
        for c in range(self.N):
            x[c, 0] = c * self.dL

        self.setState(x, u)

        return self.getState()

    def render(self, test = None):
        if self.render_mode is None:
            return

        state = self.getState()

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
            self.goal_circle = self.ax.plot(self.goal[0], self.goal[1], 'g', marker = '+', markersize=10, markeredgewidth=2)[0]

            lim = .2
            self.ax.axis('equal')
            self.ax.set_xlim(-lim, lim)
            self.ax.set_ylim(-lim, lim)
            plt.show(block=False)

            plt.draw()
            plt.pause(0.001)

        else:
            
            for n, circle in enumerate(self.circles):
                circle.set(center=(state.x_1[n], state.x_2[n]))
            self.goal_circle.set_xdata(self.goal[0])     
            self.goal_circle.set_ydata(self.goal[1])     
            plt.draw()
            plt.pause(0.001) 
            self.fig.canvas.draw()


        self.fig.canvas.draw()
        image = np.array(self.fig.canvas.renderer.buffer_rgba())
        if image.shape[0] == 750:
            self.save_render.append(image)
        print(image.shape)
        # now you have a numpy array representing the rendered image
        # print(image.shape)  # (height, width, channels)

    def close(self):
        if self.render:
            plt.close(self.fig)
            imageio.mimsave('animation.gif', self.save_render, duration=self.dt*1000)

