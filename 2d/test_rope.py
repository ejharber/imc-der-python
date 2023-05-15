from rope import RopePython
import numpy as np

rope = RopePython(True)

for _  in range(10):
	rope.reset()
	rope.step(np.array([0.1, 0.1, 1]))
