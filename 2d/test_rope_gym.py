from rope_gym import RopeEnv
import gym 
from stable_baselines3 import PPO
import numpy as np 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

# env = RopeEnv(1)

# for _ in range(500):
#     env.reset()
#     for _ in range(10):
#         action = env.action_space.sample()
#         print(action)
#         _, reward, terminate, _ = env.step(action)
#         print(reward)

#         if terminate:
#             break


# # #     # time.sleep(0.01)
env = SubprocVecEnv([lambda: RopeEnv() for i in range(20)], start_method="fork")
# # env[0].render_mode = 1
# # obs = env.reset()

# model = PPO("MlpPolicy", env,  learning_rate=0.003, n_steps=500, verbose=1, policy_kwargs={'net_arch':dict(pi=[8, 8], vf=[8, 8])})
model = PPO("MlpPolicy", env, n_steps=50, batch_size=50, verbose=1)

model.learn(total_timesteps=10_000_000, log_interval=1)
model.save("rope_soft_iter_python")


env = RopeEnv(1)
model = PPO.load("rope_soft_iter_python")

for _ in range(10):
    obs = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
        if done: break

env.close()