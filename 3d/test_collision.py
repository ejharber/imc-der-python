from rope import Rope
import numpy as np 

rope = Rope(True, 2, seed=None)
print('test')
rope.updateRender()

for i in range(10000):
    print(i)
    rope.stepVel([0, 0, 0])
    if i % 10 == 0:
        print("update")
        rope.updateRender()

# for i in range(100):
    # rope.stepVel([1, 1, 1])
    # if i % 10 == 0:
        # rope.updateRender()


# for i in range(1000):
    # rope.stepVel([-1, -1, 0])
    # if i % 10 == 0:
        # rope.updateRender()