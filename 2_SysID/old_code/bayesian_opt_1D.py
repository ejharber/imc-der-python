from rope_gym import RopeEnv
import numpy as np 
from skopt import gp_minimize
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt.plots import plot_convergence
from scipy.optimize import fsolve 

def cost(param, actions, training_states):

        env = RopeEnv()
        m = param[0]
        env.rope.m = m

        cost = 0

        ## could probably speed this up using paraellization 
        # for i, action in enumerate(actions):
        #     env.reset()
        #     success, potential_training_state = env.rope.step(action)

        #     if not success:
        #     	return 1e8

        #     cost += np.linalg.norm(training_states[i] - potential_training_state)

        for _ in range(100):

            while(True):
                i = np.random.randint(low = 0, high = len(actions))
                env.reset()
                success, potential_training_state = env.rope.step(actions[i])

                if success:
                    cost += np.linalg.norm(training_states[i] - potential_training_state)
                    break

        return cost

data = np.load("bayes_opt_training_large.npz")
actions = data["actions"]
training_states = data["training_states"]

print(cost([0.2], actions, training_states))

cost_fun = lambda x : cost(x, actions, training_states)

res = gp_minimize(cost_fun,                  # the function to minimize
                  [(1e-1, 5e-1)],      # the bounds on each dimension of x
                  n_calls=50,         # the number of evaluations of f
                  n_initial_points=10,
                  n_restarts_optimizer=5,
                  verbose=True,
                  noise=0.1**2)


for n_iter in range(5):
    # Plot true function.
    plt.subplot(5, 2, 2*n_iter+1)

    if n_iter == 0:
        show_legend = True
    else:
        show_legend = False

    ax = plot_gaussian_process(res, n_calls=n_iter*10,
                               show_legend=show_legend, show_title=False,
                               show_next_point=False, show_acq_func=False)
    ax.set_ylabel("")
    ax.set_xlabel("")
    # Plot EI(x)
    plt.subplot(5, 2, 2*n_iter+2)
    ax = plot_gaussian_process(res, n_calls=n_iter*10,
                               show_legend=show_legend, show_title=False,
                               show_mu=False, show_acq_func=True,
                               show_observations=False,
                               show_next_point=True)
    ax.set_ylabel("")
    ax.set_xlabel("")

plt.figure()

GP = res.models[-1]
x_min = res.x

cost_min, sigma = GP.predict([x_min], return_std=True)

print(res)
print(x_min)
print(cost_min, sigma)

zeros = lambda x: GP.predict([x]) - (GP.predict([x_min]) + sigma)

zeros_1 = fsolve(zeros, x_min)

print("difference", zeros_1[0] - x_min[0])
if zeros_1[0] > x_min[0]:
    print("greater than")
    zeros_2 = fsolve(zeros, [x_min[0] - 2*abs(zeros_1[0] - x_min[0])])
else:
    print("less than")
    # zeros = lambda x: if x < x_min: abs(zeros_1[0] - x_min[0] GP.predict([x]) - (GP.predict([x_min]) + sigma)

    zeros_2 = fsolve(zeros, [x_min[0] + 10*abs(x_min[0] - zeros_1[0])])

print(zeros_1, zeros_2)

plt.show()