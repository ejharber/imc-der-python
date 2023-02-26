from rope import Rope
import numpy as np 

rope = Rope(True, 10)

for i in range(10000):
    rope.stepVel([0, 0, 1])
    if i % 100 == 0:
        rope.updateRender()

for i in range(1000):
    rope.stepVel([0, 0, 2])
    if i % 10 == 0:
        rope.updateRender()