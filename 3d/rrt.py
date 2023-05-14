import numpy as np
from rope import Rope
import random 
from scipy.optimize import minimize 
import time 
from concurrent.futures import ThreadPoolExecutor, Executor
import copy 
import os 
import matplotlib.animation as animation

class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, state):
            self.state = state
            self.index = None
            self.path = []
            self.parent_index = None
            self.dead_end = False
            self.is_goal = False

    def __init__(self,
                 start_state,
                 goal_state,
                 rope,
                 goal_sample_rate=20):

        self.rope = rope
        self.rope.setState(start_state)

        self.start_state_node = self.Node(start_state)
        self.goal_state_node = self.Node(goal_state)
        self.node_list = []
        self.max_iter = 100
        self.goal_sample_rate = goal_sample_rate
        self.horizon = 1
        self.dist_to_goal = []

        # data to save
        self.from_nodes = []
        self.new_nodes = []
        self.to_nodes = []
        self.success = []

    def planning(self):

        self.node_list = [self.start_state_node]
        self.node_list[0].index = 0

        for _ in range(self.max_iter):
            # Generate random node
            rnd_node = self.get_random_node()

            # Find closest node
            nearest_ind, _ = self.get_nearest_node_index(rnd_node, rnd_node.is_goal)

            # print(len(self.node_list), nearest_ind)

            # Try to naviate towards new node
            new_node = self.stear_grad_decent(self.node_list[nearest_ind], rnd_node)

            if new_node is None:
                continue 

            # Add to list of nodes
            self.node_list.append(new_node)
            self.node_list[-1].index = len(self.node_list) - 1
            
        return self.generate_final_course()

    def stear_grad_decent(self, from_node, to_node):
        def costFunction(rope_temp, inputs, N, start_state, goal_state):

            inputs = np.zeros((3, self.horizon))            

            rope_temp.setState(start_state)
            rope_temp.stepVel(inputs) 
            cost = rope_temp.costFun(rope_temp.getState(), goal_state)
            jac = np.ones((3, 1)) * cost

            epsilon = 1e-8

            for num_steps in range(10):

                for i in range(3):
                    rope_temp.setState(start_state)
                    inputs[i, 0] += epsilon
                    success = rope_temp.stepVel(inputs)
                    if not success:
                        return None
                    next_cost = rope_temp.costFun(rope_temp.getState(), goal_state)
                    jac[i] -= next_cost
                    inputs[i, 0] -= epsilon

                # print(jac)
                # jac /= np.linalg.norm(jac)
                inputs -= .002*jac
                # print("cost", cost)
                # for i in range(3):
                #     inputs[i, 0] = min(1, inputs[i, 0])
                #     inputs[i, 0] = max(-1, inputs[i, 0])
                # print(inputs, cost)
                if np.max(abs(inputs) > 2):
                    print(np.max(abs(inputs)))

            return inputs

        inputs = np.zeros((3, self.horizon))
        if len(from_node.path) > 0:
            inputs = np.copy(from_node.path[-1])

        rope_temp = Rope(False, n_parts = self.rope.N)

        self.rope.setState(from_node.state)
        path = []

        for step in range(1000):
            inputs = costFunction(rope_temp, np.copy(inputs), self.rope.N, self.rope.getState(), to_node.state)

            if inputs is None:
                time.sleep(10)
                return None

            success = self.rope.stepVel(inputs) 

            # if step % 100 == 0:
            self.rope.updateRender()

            if not success:
                return None

            path.append(np.copy(inputs))

            new_node = self.Node(self.rope.getState())
            og_node_cost = self.rope.costFun(from_node.state, to_node.state)
            new_node_cost = self.rope.costFun(new_node.state, to_node.state)

            print(new_node_cost)

            # print("step:", og_node_cost, new_node_cost, og_node_cost > new_node_cost)

            # print("step:", next_cost)

        new_node = self.Node(self.rope.getState())
        new_node.parent_index = from_node.index
        new_node.path = path

        og_node_cost = self.rope.costFun(from_node.state, to_node.state)
        new_node_cost = self.rope.costFun(new_node.state, to_node.state)

        print("Optimization:", og_node_cost - new_node_cost, to_node.is_goal)

        if to_node.is_goal:
            from_node.dead_end = True

        self.from_nodes.append(from_node.state)
        self.new_nodes.append(new_node.state)
        self.to_nodes.append(to_node.state)

        return new_node

    def generate_final_course(self):

        min_dist = np.inf
        min_dist_i = -1

        for i, node in enumerate(self.node_list):
            dist = self.rope.costFun(node.state, self.goal_state_node.state)
            self.dist_to_goal.append(dist)
            if dist < min_dist:
                min_dist = dist
                min_dist_i = i

        node = self.node_list[min_dist_i]
        path = node.path

        while node.parent_index is not None:
            node = self.node_list[node.parent_index]
            path = node.path + path
            print("test")

        return path, self.node_list[min_dist_i].state

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd_node = self.Node(self.rope.getRandomState())
            return rnd_node
        else:  # goal point sampling
            rnd_node = self.Node(self.goal_state_node.state)
            rnd_node.is_goal = True
        return rnd_node
    
    def get_nearest_node_index(self, rnd_node, skip_dead_ends = False):

        min_index = -1
        min_cost = np.inf

        for n, node in enumerate(self.node_list):
            if node.dead_end and skip_dead_ends: continue 
            cost = self.rope.costFun(node.state, rnd_node.state)
            if cost < min_cost:
                min_cost = cost
                min_index = n

        return min_index, cost

def main():

    num_cores = 1

    for num_beads in range(11, 12):
        for seed in range(7,10):

            print(num_beads, seed)

            save_name = str(num_beads) + "_" + str(seed)
            file_name = "results"

            # if save_name in os.listdir(file_name): continue 

            rope = Rope(True, num_beads, seed)

            for _ in range(1000):
                rope.stepVel()
            # print(rope.getState().Q.shape)
            # exit()

            goal_state = Rope(False, num_beads).getState()
            start_state = rope.getState()

            rrt = RRT(
                start_state = start_state,
                goal_state = goal_state,
                rope = rope
                )

            path, final_state = rrt.planning()

            np.savez(file_name + "/" + save_name, path=path, dist_to_goal = rrt.dist_to_goal, goal_state=goal_state,
                     from_nodes = rrt.from_nodes, new_nodes = rrt.new_nodes, to_nodes = rrt.to_nodes, success = rrt.success)

            rope.close()

            rope = Rope(True, num_beads)

            rope.setState(start_state)

            Qs = []

            print(len(path))

            for i in range(len(path)):
                Q = rope.stepVel(path[i])
                Qs += Q

            if rope.costFun(final_state, rope.getState()) > 1e-9:
                
                print(final_state)
                print(rope.getState())

                rope.setState(start_state)

                for i in range(len(path)):
                    rope.stepVel(path[i])

                print(rope.getState())

                print("error line 296")
                print(rope.costFun(final_state, rope.getState()))
                # exit()

            rope.close()


if __name__ == '__main__':
    main()