import sys

from rope_gym_RL import RopeEnvRL

import gym 
from stable_baselines3 import PPO
import numpy as np 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import logger, spaces
import torch


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

        self.cnn_pos = nn.Sequential(
            layer_init(nn.Conv2d(2, 8, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(8, 16, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 2, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 4, 2, stride=1)),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        self.network = nn.Sequential(
                layer_init(nn.Linear(47 - 1, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 2 + 1)),
                )
    
    def forward(self, observations):
        def data2imgs_test(traj_pos, traj_force):
            img_pos = traj_pos
            img_pos_x = img_pos[:, ::2, :]
            img_pos_y = img_pos[:, 1::2, :]
            img_pos = torch.concatenate((img_pos_x, img_pos_y), axis=0)
            img_pos = img_pos[None, :, :, :]
            # img_pos = torch.concatenate((img_pos_x, img_pos_y), axis=0)
            # img_pos = torch.from_numpy(img_pos.astype(np.float32))

            # img_force = traj_force
            # img_force_x = torch.expand_dims(img_force[::2, :], 0)
            # img_force_y = torch.expand_dims(img_force[1::2, :], 0)
            # img_force = torch.concatenate((img_force_x, img_force_y), axis=0)
            # img_force = torch.expand_dims(img_force, 0)
            # img_force = torch.repeat(img_force, 100_000, 0)
            # img_force = torch.from_numpy(img_force.astype(np.float32))

            return img_pos, None

        img_pos, img_force = data2imgs_test(observations["traj_pos"], observations["traj_force"])
        
        goal_vec = observations["goal_vec"]
        
        print(img_pos.shape, goal_vec.shape)
        x_pos = self.cnn_pos(img_pos)
        x = torch.cat((x_pos, goal_vec), dim=1)
        print(x.shape)
        x = self.network(x)
        print(x.shape)
        return x


env = SubprocVecEnv([lambda: RopeEnvRL(True) for i in range(1)], start_method="fork")
# env = DummyVecEnv(RopeEnv(True))
# # env[0].render_mode = 1
# # obs = env.reset()
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor
)
# model = PPO("MultiInputPolicy", env,  learning_rate=0.005, n_steps=500, verbose=1, policy_kwargs=policy_kwargs)
model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

model.learn(total_timesteps=5_000_000, log_interval=1)
model.save("rope_soft_iter_python")


env = RopeEnv(1)
model = PPO.load("rope_random_delta")

for _ in range(10):
    obs = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
        if done: break

env.close()