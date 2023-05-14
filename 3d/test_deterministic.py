from rope import Rope
import numpy as np 

rope = Rope(False, 10)

for i in range(100):
    rope.stepVel([0, 0, -5.0])

rope.reset()

for i in range(100):
    rope.stepVel([0, 0, -5.0])

final_state_1 = rope.getState()

# rope = Rope(False, 10)
rope.reset()

for i in range(100):
    rope.stepVel([0, 0, -5.0])

final_state_2 = rope.getState()

# rope = Rope(False, 10)
rope.reset()

for i in range(100):
    rope.stepVel([0, 0, -5.0])

final_state_3 = rope.getState()

print(final_state_1.Q - final_state_2.Q)
print(final_state_3.Q - final_state_2.Q)
