import sys
sys.path.append("../gym/")

from rope_gym import RopeEnv
import numpy as np 

env = RopeEnv()

actions = []
training_states = []

force_scales = []
pos_scales = []


for i in range(2000):
    print(i)
    env.reset()
    action = env.action_space.sample()
    success, training_state = env.rope.step(action)

    if success:
        actions.append(action)
        training_states.append(training_state)

        force_scales.append(np.linalg.norm(training_states[-1][2222-202:]))
        pos_scales.append(np.linalg.norm(training_states[-1][:2222-202]))


print(np.min(force_scales), np.max(force_scales), np.mean(force_scales))
print(np.min(pos_scales), np.max(pos_scales), np.mean(pos_scales))

np.savez("data/training", actions = actions, training_states=training_states)
