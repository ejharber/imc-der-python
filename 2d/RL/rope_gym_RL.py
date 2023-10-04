import math
from gym import utils
import numpy as np
import gym
from gym import logger, spaces
import imageio
from rope import RopePython

import sys
sys.path.append("../Supervized/zero_shot")
from model import NeuralNetwork
import torch 

# should change this to inhearet from rope gym
class RopeEnv(gym.Env,):

    def __init__(self, random_sim_params, render_mode = None):
        
        # I should replace this with a yaml file
        input_dim = 3
        hidden_dim = 128
        output_dim = 2

        self.model = NeuralNetwork(input_dim, hidden_dim, output_dim)
        self.model.load_state_dict(torch.load("../Supervized/zero_shot/model_params/128_128", map_location=torch.device('cpu')))

        self.rope = RopePython(random_sim_params, render_mode)
        self._simulation_start_state = None

        self.original_cost = 1

        self.low = -0.2
        self.high = 0.2
        
        self.goal = np.random.random(2) * (self.high - self.low) - self.high
        self.goal_circle = None

        self.max_iter = 3
        self.current_iter = 0

        action_low = np.ones(3)
        action_low[:2] *= self.low
        action_low[2] *= -np.pi

        action_high = np.ones(3)
        action_high[:2] *= self.high
        action_high[2] *= np.pi

        self.action_space = spaces.Box(action_low, action_high, dtype=np.float64)
        self.action = np.zeros(3)

        self.goal_space = spaces.Box(action_low[:2], action_high[:2], dtype=np.float64)

        pos_traj_low = - np.ones((20, 101)) * np.inf
        pos_traj_high = np.ones((20, 101)) * np.inf        
        pos_traj_os = spaces.Box(pos_traj_low, pos_traj_high, dtype=np.float64)

        force_traj_low = - np.ones((2, 101)) * np.inf
        force_traj_high = np.ones((2, 101)) * np.inf        
        force_traj_os = spaces.Box(force_traj_low, force_traj_high, dtype=np.float64)

        self.observation_space = spaces.Dict({'pos_traj': pos_traj_os, 
                                       'force_traj': force_traj_os,
                                       'action': self.action_space, 
                                       'goal': self.goal_space})

        self.reset()

    def costFun(self):

        state = self.rope.getState()
        to_state = self.goal

        cost = (state.x_1_ee - to_state[0]) ** 2  + (state.x_2_ee - to_state[1]) ** 2
        cost = np.sqrt(cost)

        return cost

    def step(self, action):

        self.rope.setState(self._simulation_start_state)

        self.action += action

        # print(self.action)
        success, self.traj_pos, self.traj_force = self.rope.step(self.action) #this might have to be updated to use with RL

        if not success:
            observation_state = {'pos_traj': np.zeros((20,101)), 
                                 'force_traj': np.zeros((2,101)),
                                 'action': self.action, 
                                 'goal': self.goal}

            return observation_state, 0, True, {}

        observation_state = {'pos_traj': self.traj_pos, 
                             'force_traj': self.traj_force,
                             'action': self.action, 
                             'goal': self.goal}

        # observation_state = {'goal': self.goal}

        reward = (self.original_cost - self.costFun()) / self.original_cost * 100

        self.current_iter += 1
        terminate = False
        if self.current_iter >= self.max_iter:
            terminate = True

        # print(reward)
        return observation_state, reward, terminate, {}
        # return np.concatenate((self.goal, training_state)), reward, terminate, {}

    def reset(self, seed = None):

        self._simulation_start_state = self.rope.reset(seed)

        np.random.seed(seed)

        self.X = np.random.rand(3, 100_000)
        self.X[:2, :] = self.X[:2, :] * (self.high - self.low) - self.high
        self.X[2, :] = self.X[2, :] * 2 * np.pi - np.pi
        self.X = self.X.T

        self.X = torch.from_numpy(self.X.astype(np.float32))
        self.y = self.model(self.X).detach().numpy()n

        self.goal = np.random.random(2) * (self.high - self.low) - self.high
        self.render()

        self.current_iter = 0

        i = np.argmin(np.linalg.norm(self.goal - self.y, axis = 1))
        self.action = self.X[i, :].detach().numpy()

        print(self.action)
        observation_state, _, _, _ = self.step(0) #this might have to be updated to use with RL
        print("done first iter")
        self.original_cost = self.costFun()

        return observation_state

    def render(self, render_mode = None):

        if self.rope.render_mode is not None:
            if self.goal_circle is None:
                self.goal_circle = self.rope.ax.plot(self.goal[0], self.goal[1], 'g', marker = '+', markersize=6, markeredgewidth=1)[0]
            else:
                self.goal_circle.set_xdata(self.goal[0])     
                self.goal_circle.set_ydata(self.goal[1])     
            self.rope.render()


    def close(self):
        self.rope.close()


