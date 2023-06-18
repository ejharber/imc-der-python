import sys
sys.path.append("../gym/")

# from rope_gym import RopeEnv
from rope_gym2 import RopeEnv

import gym 
from stable_baselines3 import PPO
import numpy as np 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

# # #     # time.sleep(0.01)
env = SubprocVecEnv([lambda: RopeEnv() for i in range(20)], start_method="fork")

# model = PPO("MlpPolicy", env,  learning_rate=0.003, n_steps=500, verbose=1, policy_kwargs={'net_arch':dict(pi=[8, 8], vf=[8, 8])})
model = PPO("MlpPolicy", env, n_steps=50, batch_size=100, verbose=1)

model.learn(total_timesteps=2_000_000, log_interval=1)
model.save("rope_random_delta")


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