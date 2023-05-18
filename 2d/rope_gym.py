import math
from gym import utils
import numpy as np
import gym
from gym import logger, spaces
import imageio
from rope import RopePython

class RopeEnv(gym.Env,):

    def __init__(self, render_mode = None):
        
        self.rope = RopePython(render_mode)
        self._simulation_start_state = self.rope.reset()

        self.original_cost = 1

        self.low = -0.2
        self.high = 0.2
        
        self.goal = np.random.random(2) * (self.high - self.low) - self.high
        self.goal_circle = None

        self.max_iter = 1
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
        observation_low[:2] *= self.low - self.rope.length        
        observation_low[2:4] *= self.low * self.max_iter - self.rope.length
        observation_low[4:] = action_low

        observation_high = np.ones(7)
        observation_high[:2] *= self.high + self.rope.length
        observation_high[2:4] *= self.high * self.max_iter + self.rope.length
        observation_high[4:] = action_high

        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float64)

        self.reset()

    def costFun(self):

        state = self.rope.getState()
        to_state = self.goal

        cost = (state.x_1_ee - to_state[0]) ** 2  + (state.x_2_ee - to_state[1]) ** 2
        cost = np.sqrt(cost)

        return cost

    def step(self, next_action):

        self.rope.setState(self._simulation_start_state)

        self.action = next_action
        self.action[2] = (self.action[2] + np.pi) % (2*np.pi) - np.pi

        success = self.rope.step(self.action)

        if not success:
            return np.concatenate((self.goal, np.array([0, 0]), self.action)), -10000, True, {}

        reward = (self.original_cost - self.costFun()) / self.original_cost * 100

        state = self.rope.getState()

        self.current_iter += 1
        terminate = False
        if self.current_iter >= self.max_iter:
            terminate = True

        return np.concatenate((self.goal, state.x_ee, self.action)), reward, terminate, {}

    def reset(self, seed = None):

        self._simulation_start_state = self.rope.reset(seed)

        self.goal = np.random.random(2) * (self.high - self.low) - self.high

        self.action = np.zeros(3)
        self.rope.step(self.action_space.sample())

        self.current_iter = 0

        state = self.rope.getState()   
        self.original_cost = self.costFun()

        self.render()

        return np.concatenate((self.goal, state.x_ee, self.action))

    def render(self, test = None):

        if self.rope.render_mode is not None:
            if self.goal_circle is None:
                self.goal_circle = self.rope.ax.plot(self.goal[0], self.goal[1], 'g', marker = '+', markersize=6, markeredgewidth=1)[0]
            else:
                self.goal_circle.set_xdata(self.goal[0])     
                self.goal_circle.set_ydata(self.goal[1])     
            self.rope.render()


    def close(self):
        self.rope.close()


