import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
import time
import seaborn as sns
from build import pywrap

class Rope:
    """docstring for rope_sim"""

    class State:
        """
        Rope full state space
        """

        def __init__(self, x, u, d1, d2, tangent, ref_twist):
            self.x = x
            self.u = u
            self.d1 = d1
            self.d2 = d2
            self.tangent = tangent
            self.ref_twist = ref_twist
            self.Q = np.copy(np.concatenate((x, u)))

    def __init__(self, render=False, n_parts=5):
        
        self._simulation = pywrap.world("option.txt", n_parts)
        self.N = n_parts

        self.render = render
        self.fig, self.ax = None, None
        self.display_count = 0

        self.radius = 0.0046 # should get these from sim
        self.space_length = 0.01 # should get these from sim

        self.reset()

    def reset(self):

        self._simulation.setupSimulation()
        self.updateRender()

    def setState(self, state):

        x = state.x
        u = state.u
        d1 = state.d1
        d2 = state.d2
        tangent = state.tangent
        ref_twist = state.ref_twist

        self._simulation.setStatePos(x)
        self._simulation.setStateVel(u)
        self._simulation.setStateD1(d1)
        self._simulation.setStateD2(d2)
        self._simulation.setStateTangent(tangent)
        self._simulation.setStateRefTwist(ref_twist)

        self.updateRender()

    def getState(self):

        x = self._simulation.getStatePos()
        u = self._simulation.getStateVel()
        d1 = self._simulation.getStateD1()
        d2 = self._simulation.getStateD2()
        tangent = self._simulation.getStateTangent()
        ref_twist = self._simulation.getStateRefTwist()

        state = self.State(x, u, d1, d2, tangent, ref_twist)
        
        return state 

    def stepVel(self, u = None):

        if u is not None:
            self._simulation.setPointVel(u)

        self._simulation.stepSimulation()

    def updateRender(self, x = None):

        if self.render:
            state = self.getState()

            q = state.Q[:4*self.N]

            if x is None:
                q_x = q[0::4]
                q_y = q[2::4]
            else:
                q_x = x[0::4]
                q_y = x[2::4]

            # print(q[1::4])

            if self.fig is None:
                sns.set() # Setting seaborn as default style even if use only matplotlib
                self.fig, self.ax = plt.subplots(figsize=(15, 15))
                # self.fig.figsize(30, 30)

                self.circles = []
                self.circles += [patches.Circle((q_x[0], q_y[0]), radius=self.radius, color='red')]
                self.ax.add_patch(self.circles[-1])
                for n in range(1, self.N):
                    self.circles += [patches.Circle((q_x[n], q_y[n]), radius=self.radius, color='blue')]
                    self.ax.add_patch(self.circles[-1])
                lim = .5
                self.ax.axis('equal')
                self.ax.set_xlim(-lim, lim)
                self.ax.set_ylim(-lim, lim)

                plt.show(False)

            else:
                
                for n, circle in enumerate(self.circles):
                    circle.set(center=(q_x[n], q_y[n]))
                self.fig.canvas.draw()
                self.display_count = 0

    def getRandomState(self, seed = None):
        def add_next_state(state):
            if len(state) // 2 == self.N:
                return state

            for num_trial in range(100):
                # Step 1: Generate a random position for the next bead
                x_last, y_last = state[-1]

                theta = np.random.uniform(0, 2*np.pi)

                x = x_last + self.space_length*np.cos(theta)
                y = y_last + self.space_length*np.sin(theta)

                # Step 2: Check Collision for next bead
                collision = False
                for (x_prev, y_prev) in state:
                    if (x_prev - x) ** 2 + (y_prev - y) ** 2 < (2*self.radius) ** 2: # if collissions :
                        collision = True
                        break
                if collision:
                    continue   

                # Step 3: Try next bead 
                next_state = add_next_state(state + [(x, y)])

                if len(next_state) > 0:
                    return next_state

            return []

        # Step 0: Generate initial bead position 
        np.random.seed(seed)
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        initial_state = [(x, y)]

        # Step 3: Add next bead
        state = add_next_state(initial_state)
        full_state = []
        for (x, y) in state:
            full_state += [x, 0, y, 0] # pad state with z and theta

        return full_state

    def close(self):
        if self.render:
            plt.close(self.fig)
