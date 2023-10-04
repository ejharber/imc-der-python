import sys
sys.path.append("../gym/")

# from rope_gym import RopeEnv
from rope_gym import RopeEnv

import gym 
from stable_baselines3 import PPO
import numpy as np 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import logger, spaces
import torch as th


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        extractor = nn.Sequential(nn.Linear(7, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 3))

        self.extractor = extractor
        self._features_dim = 3

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        # for key, extractor in self.extractors.items():
            # encoded_tensor_list.append(extractor(observations[key]))
        # feedback = th.cat(observations["goal"], observations["pos_traj"][-2:,-1]) 
        # encoded_tensor_list.append(extractor(feedback))
        feedback = [observations["goal"], observations["pos_traj"][:,-2:,-1], observations["action"]]
        feedback = th.cat(feedback, dim=1)
        # print(feedback.shape)
        # feedback.cat(observations["pos_traj"][-2:,-1])
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return self.extractor(feedback)


env = SubprocVecEnv([lambda: RopeEnv(True) for i in range(1)], start_method="fork")
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