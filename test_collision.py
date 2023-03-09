from rope import Rope
import numpy as np 

rope = Rope(True, 20)

for i in range(1000):
    rope.stepVel([0, 0, 2])
    if i % 10 == 0:
        rope.updateRender()