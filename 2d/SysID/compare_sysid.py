import sys
sys.path.append("../gym/")

from rope_gym import RopeEnv

from skopt import dump, load
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

sns.set()

def cost_fun(params, data, actions, training_states):

    EI = params[0] 
    EA = params[1]
    damp = params[2]
    m = params[3]

    env = RopeEnv(False)

    env.rope.EI = EI
    env.rope.EA = EA
    env.rope.damp = damp
    env.rope.m = m

    cost = 0

    for _ in range(500):

        while(True):
            i = np.random.randint(low = 0, high = len(actions))
            env.reset()
            success, potential_training_state = env.rope.step(actions[i])

            if success:
                cost += np.linalg.norm(training_states[i][2222-202-2:2222-202] - potential_training_state[2222-202-2:2222-202]) 
                break

    return cost


def get_costs_BO(res):

    data = np.load("data/bayes_opt_training_large.npz")
    actions = data["actions"]
    training_states = data["training_states"]

    costs = []
    for i in range(0, len(res.x_iters), 5):

        x = res.x_iters[i]
        cost = cost_fun(x, data, actions, training_states)

        if len(costs) == 0:
            costs.append(cost)
            continue         

        if costs[-1] < cost:
            costs.append(costs[-1])
        else:
            costs.append(cost)

    return costs

def get_costs_DE(x_iters):

    data = np.load("data/bayes_opt_training_large.npz")
    actions = data["actions"]
    training_states = data["training_states"]

    costs = []
    for i in range(1, len(x_iters), 5):

        x = x_iters[i]
        cost = cost_fun(x, data, actions, training_states)
        # print(i, cost)
        if len(costs) == 0:
            costs.append(cost)
            continue         

        if costs[-1] < cost:
            costs.append(costs[-1])
        else:
            costs.append(cost)

    return costs

plt.figure()
# costs_BO_force = get_costs_BO(load("BO_force.pkl"))
# costs_BO_no_force = get_costs_BO(load("BO_no_force.pkl"))
# costs_BO_force_scaled = get_costs_BO(load("BO_force_scaled.pkl"))
# costs_BO_no_force_scaled = get_costs_BO(load("BO_no_force_scaled.pkl"))

# plt.plot(costs_BO_force, label="BO: force")
# plt.plot(costs_BO_no_force, label="BO: no force")
# plt.plot(costs_BO_force_scaled, label="BO: force scaled")
# plt.plot(costs_BO_no_force_scaled, label="BO: no force scaled")
# plt.legend()


# plt.figure()
# costs_DE_force = get_costs_DE(np.load("DE_force.npy"))
# costs_DE_no_force = get_costs_DE(np.load("DE_no_force.npy"))
costs_DE_force_scaled = get_costs_DE(np.load("data/DE_force_scaled_noise_500samples.npy"))
costs_DE_no_force_scaled = get_costs_DE(np.load("data/DE_no_force_scaled_noise_500samples.npy"))

# plt.plot(costs_DE_force, label="DE: force")
# plt.plot(costs_DE_no_force, label="DE: no force")
plt.plot(costs_DE_force_scaled, label="DE: force scaled")
plt.plot(costs_DE_no_force_scaled, label="DE: no force scaled")
plt.legend()

plt.show()





