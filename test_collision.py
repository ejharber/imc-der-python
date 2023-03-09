from rope import Rope
import numpy as np 

rope = Rope(True, 2)

for i in range(10000):
    rope.stepVel([-.1, 0, 0])
    if i % 100 == 0:
        rope.updateRender()
print(rope.getState().x)
print(rope.getState().u)
print(rope.getState().d1)
print(rope.getState().d2)
print(rope.getState().tangent)
print(rope.getState().ref_twist)


for i in range(1000):
    rope.stepVel([0, 0, 2])
    if i % 10 == 0:
        rope.updateRender()