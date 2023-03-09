from rope import Rope
import numpy as np 
import time

rope = Rope(True, 10)

for i in range(100000):
    rope.stepVel([.1, 0, 0])
    if i % 100 == 0:
        rope.updateRender()
    # time.sleep(0.01)
