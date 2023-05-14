from rope import Rope
import numpy as np 
import time

rope = Rope(True, 10)
rope.updateRender()

for i in range(100000):
    # print(i)
    rope.stepVel([0, 0, 0])
    rope.updateRender()
