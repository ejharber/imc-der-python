import sys
sys.path.append('../gym/')
import rope_gym_registration
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

# Parallel environments
vec_env = make_vec_env("RopeEnv-v0", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25)

