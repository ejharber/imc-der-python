from rope import Rope
import numpy as np 

rope = Rope(False, 10)

for i in range(100):
    rope.stepVel([0, 0, -5])

start_state = rope.getState()

for i in range(100):
    rope.stepVel([0, 0, 2])

final_state_1 = rope.getState().Q

rope.setState(start_state)

for i in range(100):
    rope.stepVel([0, 0, 2])

final_state_2 = rope.getState().Q

print(final_state_1 - final_state_2)

