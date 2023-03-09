from rope import Rope
import numpy as np 
import time


rope = Rope(True, 2)

for _ in range(100):
   
    rope.stepVel([1, 0, 0])
  
    rope.updateRender()
    time.sleep(.01)

for _ in range(100):
   
    rope.stepVel([0, 1, 0])
  
    rope.updateRender()
    time.sleep(.01)

for _ in range(100):
   
    rope.stepVel([0, 0, 1])
  
    rope.updateRender()
    time.sleep(.01)

rope.close()
