import math
from gym import utils
import numpy as np
import gym
from gym import logger, spaces
import imageio
from rope import RopePython

import sys

class RopeEnv(gym.Env,):

    def __init__(self, random_sim_params, render_mode = None):
        
        self.rope = RopePython(random_sim_params, render_mode)
        self._simulation_start_state = None

        self.original_cost = 1

        self.low = -0.2
        self.high = 0.2
        
        self.goal = np.random.random(2) * (self.high - self.low) - self.high
        self.goal_circle = None

        self.max_iter = 3
        self.current_iter = 0

        self.action_low = np.ones(3)
        self.action_low[:2] *= self.low
        self.action_low[2] *= -np.pi

        self.action_high = np.ones(3)
        self.action_high[:2] *= self.high
        self.action_high[2] *= np.pi

        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float64)
        self.action = np.zeros(3)

        self.goal_space = spaces.Box(self.action_low[:2], self.action_high[:2], dtype=np.float64)

        pos_traj_low = - np.ones((40, 101)) * np.inf
        pos_traj_high = np.ones((40, 101)) * np.inf        
        pos_traj_os = spaces.Box(pos_traj_low, pos_traj_high, dtype=np.float64)

        force_traj_low = - np.ones((2, 101)) * np.inf
        force_traj_high = np.ones((2, 101)) * np.inf        
        force_traj_os = spaces.Box(force_traj_low, force_traj_high, dtype=np.float64)

        self.observation_space = spaces.Dict({'traj_pos': pos_traj_os, 
                                       'traj_force': force_traj_os,
                                       'action': self.action_space, 
                                       'goal': self.goal_space, 
                                       'goal_vec': self.goal_space})

        # self.reset()

    def costFun(self):

        state = self.rope.getState()
        to_state = self.goal

        cost = (state.x_1_ee - to_state[0]) ** 2  + (state.x_2_ee - to_state[1]) ** 2
        cost = np.sqrt(cost)

        return cost

    def step(self, action):

        self.rope.setState(self._simulation_start_state)

        self.action = action

        success, self.traj_pos, self.traj_force = self.rope.step(self.action) #this might have to be updated to use with RL

        if not success:
            observation_state = {'pos_traj': np.zeros((40,101)), 
                                 'force_traj': np.zeros((2,101)),
                                 'action': self.action, 
                                 'goal': self.goal}

            return observation_state, 0, True, {}

        observation_state = {'pos_traj': self.traj_pos, 
                             'force_traj': self.traj_force,
                             'action': self.action, 
                             'goal': self.goal}


        reward = (self.original_cost - self.costFun()) / self.original_cost * 100

        self.current_iter += 1
        terminate = False
        if self.current_iter >= self.max_iter:
            terminate = True

        self.render()

        return observation_state, reward, terminate, {}

    def reset(self, seed = None):

        render_mode = self.rope.render_mode
        self.rope.render_mode = None

        self._simulation_start_state = self.rope.reset(seed)
        self.action_space = spaces.Box(self.action_low, self.action_high, dtype=np.float64, seed = seed)

        np.random.seed(seed)

        self.goal = np.random.random(2) * (self.high - self.low) - self.high

        self.current_iter = 0
        self.action = np.zeros(3)

        observation_state, _, _, _ = self.step(self.action)

        self.original_cost = self.costFun()

        # print(self.goal_circle)

        self.rope.render_mode = render_mode

        self.render()

        return observation_state

    def render(self, render_mode = None):

        if self.rope.ax is not None:
            if self.goal_circle is None:
                if self.rope.render_mode == "Both":
                    print(self.goal)
                    self.goal_circle = self.rope.ax["A"].plot(self.goal[0], self.goal[1], 'g', marker = '+', markersize=6, markeredgewidth=1)[0]
                if self.rope.render_mode == "Human":
                    self.goal_circle = self.rope.ax.plot(self.goal[0], self.goal[1], 'g', marker = '+', markersize=6, markeredgewidth=1)[0]                    
            else:
                self.goal_circle.set_xdata(self.goal[0])     
                self.goal_circle.set_ydata(self.goal[1])     
            self.rope.render()


    def close(self):
        self.rope.close()


