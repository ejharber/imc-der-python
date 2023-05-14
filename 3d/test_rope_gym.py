import logging
from gym.envs.registration import register
from rope_gym import RopeEnv
import gym 
from stable_baselines3 import PPO
import numpy as np 
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

# env.render_mode = True

# for i in range(1000):
#     env.reset()

# for step in range(200):
    # env.render()
    # env.step(env.action_space.sample())

# env.close()
# low = -1
# high = 1
# state = np.random.uniform(low=low, high=high, size=(4,))
# print(np.array(state, dtype=np.float32).shape)

# gym.envs.register(
#      id='Rope-v0',
#      entry_point='rope_gym:RopeEnv'
# )

# env = gym.make('Rope-v0')

# # env = SubprocVecEnv([lambda: RopeEnv() for i in range(1)], start_method="fork")
# check_env(env)
# env = RopeEnv()

# for _ in range(500):
#     env.reset()
#     action = env.action_space.sample()
#     for _ in range(10):
#         print(action)
#         env.step(action)
# env._simulation.stepSimulation()
# env.render()


# # #     # time.sleep(0.01)
# env = SubprocVecEnv([lambda: RopeEnv() for i in range(20)], start_method="fork")
# # # env[0].render_mode = 1
# # # obs = env.reset()

# # model = PPO("MlpPolicy", env,  learning_rate=0.003, n_steps=500, verbose=1, policy_kwargs={'net_arch':dict(pi=[8, 8], vf=[8, 8])})
# model = PPO("MlpPolicy", env, n_steps=50, batch_size=50, verbose=1)

# model.learn(total_timesteps=10_000_000, log_interval=1)
# model.save("rope_soft_iter")


env = RopeEnv(1)
model = PPO.load("rope_soft_iter")

for _ in range(10):
    obs = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
        if done: break

env.close()
