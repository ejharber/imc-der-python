from rope import Rope
import numpy as np 
import time


for i in range(10):
# seed = 1
    rope = Rope(True, 11, i)

    for j in range(100):
       
        rope.stepVel([1, 1, 20])
        # if j % 10 == 0:
        # rope.updateRender(rope.getRandomState(seed))
        rope.updateRender()
        # time.sleep(2)

    rope.close()
