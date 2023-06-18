from rope_gym import RopeEnv
import numpy as np 
from skopt import gp_minimize
import matplotlib.pyplot as plt

def compare_params(original_params, random_params):

    env = RopeEnv()
    action = env.action_space.sample()

    env.rope.EI = original_params[0]
    env.rope.EA = original_params[1]
    env.rope.damp = original_params[2]
    env.rope.m = original_params[3]

    env.reset()

    success, training_state_original = env.rope.step(action)

    if not success:
        return np.inf

    env = RopeEnv()
    # action = env.action_space.sample() # use the same action

    env.rope.EI = random_params[0]
    env.rope.EA = random_params[1]
    env.rope.damp = random_params[2]
    env.rope.m = random_params[3]

    env.reset()

    success, training_state_random = env.rope.step(action)

    if not success:
        return np.inf

    return np.linalg.norm(training_state_original - training_state_random)



original_params = [1e-2, 1e7, 0.15, 0.2]

boundaries = [(1e-4, 1e-1), (1e6, 1e8), (1e-1, 1), (1e-1, 1)]

random_params = []

for (low, high) in boundaries:
    random_param = np.random.random(1) * (high - low) - low
    random_params.append(random_param[0])


n = 1000

cost_distribution = np.zeros(n)
running_average = np.zeros(n)

for i in range(n):

    print(i)

    cost = np.inf

    while np.isinf(cost):
        cost = compare_params(original_params, random_params)
        cost_distribution[i] = cost
        running_average[i] = np.sum(cost_distribution) / (i+1)

plt.figure()
plt.plot(running_average)


plt.show()

