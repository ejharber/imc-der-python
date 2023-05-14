from rope import Rope
import numpy as np 

rope_1 = Rope(False, 10)
rope_2 = Rope(False, 10)

for i in range(1000):
    rope_1.stepVel([0, 0, -5.0])

final_state_1 = rope_1.getState()
final_state_2 = rope_2.getState()

print(final_state_1.Q - final_state_2.Q)

rope = Rope(False, 10)

for i in range(1000):
    rope.stepVel([0, 0, -5.0])

final_state_1 = rope.getState()

rope = Rope(False, 10)

final_state_2 = rope.getState()

print(final_state_1.Q - final_state_2.Q)
