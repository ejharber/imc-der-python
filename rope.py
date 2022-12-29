import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
import time
import seaborn as sns
from build import pywrap

class Rope(object):
    """docstring for rope_sim"""
    def __init__(self, render=False, n_parts=5):
        
        self._simulation = pywrap.world("option.txt", n_parts)
        self.N = n_parts
        self.Q = None

        self.render = render
        self.fig, self.ax = None, None
        self.display_count = 0

        self.radius = 0.01

        self.reset()

    def reset(self):

        self._simulation.setupSimulation()
        self.Q = self.getState()

        self.updateRender()

    def setState(self, Q):

        self.Q = Q

        q = Q[:3*self.N]
        u = Q[3*self.N:]

        self._simulation.setStatePos(q)
        self._simulation.setStateVel(u)

        self.updateRender()

    def getState(self):
        q = self._simulation.getStatePos()
        u = self._simulation.getStateVel()
        self.Q = np.concatenate((q, u))
        # print(self.Q)
        return self.Q

    def stepVel(self, u = None):

        if u is not None:
            self._simulation.setPointVel(u)

        self._simulation.stepSimulation()

    def updateRender(self):

        if self.render:
            self.Q = self.getState()

            q = self.Q[:4*self.N]
            q_x = q[0::4]
            q_y = q[2::4]

            print(q[1::4])

            if self.fig is None:
                sns.set() # Setting seaborn as default style even if use only matplotlib
                self.fig, self.ax = plt.subplots()

                self.circles = []
                self.circles += [patches.Circle((q_x[0], q_y[0]), radius=self.radius, color='red')]
                self.ax.add_patch(self.circles[-1])
                for n in range(1, self.N):
                    self.circles += [patches.Circle((q_x[n], q_y[n]), radius=self.radius, color='blue')]
                    self.ax.add_patch(self.circles[-1])
                lim = 1
                self.ax.axis('equal')
                self.ax.set_xlim(-lim, lim)
                self.ax.set_ylim(-lim, lim)

                plt.show(False)

            else:
                
                for n, circle in enumerate(self.circles):
                    circle.set(center=(q_x[n], q_y[n]))
                self.fig.canvas.draw()
                self.display_count = 0

    # def costFun(self, from_state, to_state):

    #     pose_x = 2
    #     pose_y = 2
    #     vel_x = .01
    #     vel_y = .01

    #     Q = []

    #     for i in range(self.N):
    #         Q += [pose_x*(i+1), pose_y**(i+1)]

    #     for i in range(self.N):
    #         Q += [vel_x*(i+1), vel_y*(i+1)]

    #     Q = np.diag(Q)

    #     cost = (from_state - to_state).T @ Q @ (from_state - to_state)

    #     assert from_state.shape[0] == to_state.shape[0]
    #     assert from_state.shape[1] == to_state.shape[1]
    #     assert cost.shape[0] == 1
    #     assert cost.shape[1] == 1

    #     return cost[0,0]

    # def getRandomState(self, seed = None):
    #     def add_next_state(state):
    #         if len(state) // 2 == self.N:
    #             return state

    #         for num_trial in range(100):
    #             # Step 1: Generate a random position for the next bead
    #             theta = np.random.uniform(0, 2*np.pi)

    #             x = state[-2] + self.space_length*np.cos(theta)
    #             y = state[-1] + self.space_length*np.sin(theta)

    #             # Step 2: Check Collision for next bead
    #             collision = False
    #             for bead in range(0, len(state), 2):
    #                 if (state[bead] - x) ** 2 + (state[bead+1] - y) ** 2 < (2*self.radius) ** 2: # if collissions :
    #                     collision = True
    #                     break
    #             if collision:
    #                 continue   

    #             # Step 3: Try next bead 
    #             next_state = add_next_state(state + [x, y])

    #             if len(next_state) > 0:
    #                 return next_state

    #         return []

    #     # Step 0: Generate initial bead position 
    #     np.random.seed(seed)
    #     x = np.random.uniform(-1, 1)
    #     y = np.random.uniform(-1, 1)
    #     initial_state = [x, y]

    #     # Step 3: Add next bead
    #     state = np.array([add_next_state(initial_state)]).T

    #     return np.concatenate((state, np.zeros(state.shape)))    

    def close(self):
        if self.render:
            plt.close(self.fig)
