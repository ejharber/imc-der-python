from rope import Rope
import numpy as np 
import time

rope = Rope(True, 11)
print(rope.getState().x)
for i in range(100):
    x = rope.getRandomState()
    rope.updateRender(x)
    time.sleep(0.2)
