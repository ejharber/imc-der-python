"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union
from gym import utils
import numpy as np
from build import pywrap
import gym
from gym import logger, spaces
# from gym.envs.classic_control import utils
# from gym.error import DependencyNotInstalled
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
import time
import seaborn as sns
from build import pywrap
import copy 
from scipy.interpolate import PchipInterpolator, CubicSpline
import imageio

class RopeEnv(gym.Env,):

    class State:
        """
        Rope full state space
        """

        def __init__(self, X, U, D1, D2, Tangent, Ref_Twist):
            self.X = np.array(X)
            self.U = np.array(U)
            self.D1 = D1
            self.D2 = D2
            self.Tangent = Tangent
            self.Ref_Twist = Ref_Twist

            self.Q = np.copy(np.concatenate((self.X, self.U)))

            self.x = self.X[::2]
            self.u = self.U[::2]
            self.q = np.concatenate((self.x, self.u))

            self.x_1 = self.x[0::2]
            self.x_2 = self.x[1::2]
            self.x_1_ee = self.x_1[-1]
            self.x_2_ee = self.x_2[-1]
            self.x_ee = np.array([self.x_1_ee, self.x_2_ee])

    def __init__(self, render_mode: Optional[str] = None):
        
        self.N = 10

        self.render_mode = render_mode
        self.fig, self.ax = None, None
        self.display_count = 0

        self.radius = 0.0046 # should get these from sim
        self.space_length = 0.0046*2.1 # should get these from sim

        self.length = (self. N - 1) * self.space_length

        poses = np.zeros((3, self.N)).T
        for i in range(1, self.N):
            poses[i, 2] = poses[i-1, 2] - self.space_length

        self._simulation = pywrap.world("option.txt", np.copy(poses), np.copy(poses), self.radius)        
        self._simulation.setupSimulation()
        self._simulation_start_state = self.getState()

        self.num_steps = 0
        self.original_cost = 1

        self.low = -0.2
        self.high = 0.2
        self.goal = np.random.random(2) * (self.high - self.low) - self.high
        self.max_iter = 5
        self.current_iter = 0

        action_low = np.ones(3)
        action_low[:2] *= self.low
        action_low[2] *= -np.pi

        action_high = np.ones(3)
        action_high[:2] *= self.high
        action_high[2] *= np.pi

        self.action_space = spaces.Box(action_low, action_high, dtype=np.float64)
        self.action = np.zeros(3)

        observation_low = np.ones(7)
        observation_low[:2] *= self.low - self.length        
        observation_low[2:4] *= self.low * self.max_iter - self.length
        observation_low[4:] = action_low

        observation_high = np.ones(7)
        observation_high[:2] *= self.high + self.length
        observation_high[2:4] *= self.high * self.max_iter + self.length
        observation_high[4:] = action_high

        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float64)

        self.save_render = []

        self.reset()

    def setState(self, state):

        X = state.X
        U = state.U
        D1 = state.D1
        D2 = state.D2
        Tangent = state.Tangent
        Ref_Twist = state.Ref_Twist

        self._simulation.setStatePos(X)
        self._simulation.setStateVel(U)
        self._simulation.setStateD1(D1)
        self._simulation.setStateD2(D2)
        self._simulation.setStateTangent(Tangent)
        self._simulation.setStateRefTwist(Ref_Twist)

        self.render()

    def getState(self):

        X = self._simulation.getStatePos()
        U = self._simulation.getStateVel()
        D1 = self._simulation.getStateD1()
        D2 = self._simulation.getStateD2()
        Tangent = self._simulation.getStateTangent()
        Ref_Twist = self._simulation.getStateRefTwist()

        state = self.State(X, U, D1, D2, Tangent, Ref_Twist)
        
        return state 

    def costFun(self):

        state = self.getState()
        to_state = self.goal

        cost = (state.x_1_ee - to_state[0]) ** 2  + (state.x_2_ee - to_state[1]) ** 2
        cost = np.sqrt(cost)

        return cost

    def step(self, next_action):

        self.setState(self._simulation_start_state)

        self.action = next_action
        self.action[2] = (self.action[2] + np.pi) % (2*np.pi) - np.pi

        traj_velocity = []

        time = np.linalg.norm(self.action[:2])*2.5
        time = max(round(time, 2), 0.1)

        for dim in range(self.action.shape[0]):
            x_traj = [0, time]
            y_traj = [0, self.action[dim]]
            f_prime = CubicSpline(x_traj, y_traj, bc_type='clamped').derivative(1)
            traj_velocity.append(f_prime(np.linspace(0, time, round(time/0.01) + 1)))

        traj_velocity = np.array(traj_velocity)

        for i in range(traj_velocity.shape[1]):
            self._simulation.setPointVel(traj_velocity[:, i])
            success = self._simulation.stepSimulation()
            self.render()

            if not success:

                poses = np.zeros((3, self.N)).T
                for i in range(1, self.N):
                    poses[i, 2] = poses[i-1, 2] - self.space_length

                self._simulation = pywrap.world("option.txt", np.copy(poses), np.copy(poses), self.radius)        
                self._simulation.setupSimulation()
                self._simulation_start_state = self.getState()
        
                print(self.action, np.linalg.norm(self.action))
                return np.concatenate((self.goal, np.array([0, 0]), self.action)), -10000, True, {}

        reward = (self.original_cost - self.costFun()) / self.original_cost * 100
        
        state = self.getState()   

        self.current_iter += 1
        terminate = False
        if self.current_iter >= self.max_iter:
            terminate = True

        return np.concatenate((self.goal, state.x_ee, self.action)), reward, terminate, {}

    def reset(self, seed: Optional[int] = None):

        self.setState(self._simulation_start_state)
        self.goal = np.random.random(2) * (self.high - self.low) - self.high

        self.action = np.zeros(3)
        self.step(self.action_space.sample())
        self.current_iter = 0

        state = self.getState()   
        self.original_cost = self.costFun()

        return np.concatenate((self.goal, state.x_ee, self.action))

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
            imageio.mimsave('animation.gif', self.save_render, duration=0.01*1000)

