import math
import numpy as np

import sys
sys.path.append("../Supervized/zero_shot")
sys.path.append("../gym/")
from rope_gym import RopeEnv
from model import NeuralNetwork
from gym import spaces
import torch 

# should change this to inhearet from rope gym
class RopeEnvRL(RopeEnv):

    def __init__(self, random_sim_params, render_mode = None):
        
        super(RopeEnvRL, self).__init__(random_sim_params, render_mode)
        # I should replace this with a yaml file
        input_dim = 3
        hidden_dim = 256
        output_dim = 2

        self.model = NeuralNetwork(input_dim, hidden_dim, output_dim)
        self.model.load_state_dict(torch.load("../Supervized/zero_shot/model_params/256_256", map_location=torch.device('cpu')))

        self.original_cost = 1

        self.max_iter = 5
        self.current_iter = 0

        action_low = np.ones(3)
        action_low[:2] *= self.low / 100
        action_low[2] *= -np.pi / 100

        action_high = np.ones(3) 
        action_high[:2] *= self.high / 100
        action_high[2] *= np.pi / 100

        self.action_space = spaces.Box(action_low, action_high, dtype=np.float64)
        self.action = np.zeros(3)

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
        self.y = self.model(self.X).detach().numpy()

        self.goal = np.random.random(2) * (self.high - self.low) - self.high
        self.render()

        self.current_iter = 0

        i = np.argmin(np.linalg.norm(self.goal - self.y, axis = 1))
        self.action = self.X[i, :].detach().numpy()

        print(self.action)
        observation_state, _, _, _ = self.step(np.zeros((3))) #this might have to be updated to use with RL
        print("done first iter")
        self.original_cost = self.costFun()

        return observation_state


