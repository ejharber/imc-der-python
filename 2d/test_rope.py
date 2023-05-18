from rope import RopePython
import numpy as np

rope = RopePython(True)

rope.reset()
force, q, u, _ = rope.step(np.array([.2, 0.2, np.pi/2]))

print(q.shape)
