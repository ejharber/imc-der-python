import sys
sys.path.append("../gym/")

# from rope_gym import RopeEnv
from rope_gym2 import RopeEnv

import gym 
from stable_baselines3 import PPO
import numpy as np 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

<<<<<<< HEAD:2d/RL/train_rope_gym.py
# # #     # time.sleep(0.01)
env = SubprocVecEnv([lambda: RopeEnv() for i in range(20)], start_method="fork")
=======
env = RopeEnv(1)

for _ in range(500):
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        print(action)
        _, reward, terminate, _ = env.step(action)
        print(reward)

        if terminate:
            break


# # #     # time.sleep(0.01)
env = SubprocVecEnv([lambda: RopeEnv(1) for i in range(1)], start_method="fork")
# # env[0].render_mode = 1
# # obs = env.reset()
>>>>>>> 57dc806e3c82597fca71089224fbd9aa6d2b16b2:2d/test_rope_gym.py

# model = PPO("MlpPolicy", env,  learning_rate=0.003, n_steps=500, verbose=1, policy_kwargs={'net_arch':dict(pi=[8, 8], vf=[8, 8])})
model = PPO("MlpPolicy", env, n_steps=50, batch_size=100, verbose=1)

<<<<<<< HEAD:2d/RL/train_rope_gym.py
model.learn(total_timesteps=2_000_000, log_interval=1)
model.save("rope_random_delta")
=======
model.learn(total_timesteps=5_000_000, log_interval=1)
model.save("rope_soft_iter_python")
>>>>>>> 57dc806e3c82597fca71089224fbd9aa6d2b16b2:2d/test_rope_gym.py


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