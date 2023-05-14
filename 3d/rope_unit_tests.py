from rope import Rope
import time
import numpy as np 
from scipy.optimize import minimize 

def velocity_test():

    rope = Rope(render=True, n_parts = 10)

    tic = time.time()
    for _ in range(1000):
        rope.step_vel([0, 0, 2])

    # for _ in range(100):
        # rope.step_vel([-1, -1, 0])

    print(time.time()-tic, 500*rope.dt)

    print("done")

def random_state_test():

    rope = Rope(render=True, n_parts = 10)

    Q = rope.current_state()
    # print(.shape)

    for _ in range(20):
        Q = rope.get_random_state()
        # print(Q)
        # exit()
        rope.set_state(Q)
        time.sleep(1)

    print("done")

def mpc_test():
    def cost_function(inputs, N, start_state, goal_state, horizon):
            
        rope = Rope(n_parts = N)
        inputs = np.reshape(inputs, (3, horizon))

        rope.set_state(start_state)

        for i in range(horizon):
            rope.step_vel(inputs[:,i])
        
        cost = rope.cost_fun(rope.current_state(), goal_state)

        return cost 

    rope = Rope(n_parts = 5)
    to_state = rope.current_state()
    from_state = rope.get_random_state()

    horizon = 1

    path = []
    inputs = np.zeros((3, horizon))

    min_vel = -1
    max_vel = 1

    min_cost = np.inf
    min_out = None

    inputs_random = np.random.rand(3, horizon) * 0.0

    for trial in range(5):
        fun = lambda inputs_opt: cost_function(inputs_opt, rope.N, from_state, to_state, horizon)

        # bounds on the change in force being applied as suggest by paper 
        bounds = [(min_vel, max_vel) for _ in range(horizon*3)]
        # out = minimize(fun, inputs_random, method = 'SLSQP', jac=True, bounds=bounds)
        out = minimize(fun, inputs_random, method = 'SLSQP', bounds=bounds)

        if out.fun < min_cost:
            min_cost = out.fun
            min_out = out

        print("done early", trial, min_out.fun, rope.cost_fun(from_state, to_state))

        inputs_random = np.random.rand(3, horizon) * 2 * max_vel - max_vel

def grad_desent_test():
    def cost_function(inputs, N, start_state, goal_state):
            
        rope = Rope(n_parts = N)

        for num_steps in range(5):
            rope.set_state(start_state)
            rope.step_vel(inputs) 
            cost = rope.cost_fun(rope.current_state(), goal_state)
            jac = np.ones((3, 1)) * cost

            epsilon = 1e-8

            for i in range(3):
                rope.set_state(start_state)
                inputs[i, 0] += epsilon
                rope.step_vel(inputs) 
                next_cost = rope.cost_fun(rope.current_state(), goal_state)
                jac[i] -= next_cost
                inputs[i, 0] -= epsilon

            # print(jac)
            inputs += 100000*jac

        rope.set_state(start_state)
        rope.step_vel(inputs) 
        next_cost = rope.cost_fun(rope.current_state(), goal_state)

        # print(cost, next_cost)

        return inputs, next_cost

    rope = Rope(render=True, n_parts = 5)
    to_state = rope.current_state()
    from_state = rope.get_random_state()
    rope.set_state(from_state)

    horizon = 1

    path = []
    inputs = np.zeros((3, horizon))

    min_vel = -1
    max_vel = 1

    min_cost = np.inf
    min_out = None

    inputs_random = np.random.rand(3, horizon) * 0.0

    for trial in range(5000):
        inputs, next_cost = cost_function(inputs_random, rope.N, rope.current_state(), to_state)

        if next_cost < min_cost:
            min_cost = next_cost
            # min_out = inputs

        print(rope.cost_fun(rope.current_state(), to_state))

        rope.step_vel(inputs) 

if __name__ == '__main__':
    velocity_test()
    # random_state_test()
    # mpc_test()
    # grad_desent_test()