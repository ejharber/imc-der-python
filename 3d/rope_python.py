import functions as funcs
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Process
import time
import os
from datetime import datetime
from numba import jit


class RopePython(object):

    class State:
        """
        Rope full state space
        """
        def __init__(self, x, u):
            
            self.x = self.x
            self.u = self.u
            self.q = np.concatenate((self.x, self.u))

            self.x_1 = self.x[0::2]
            self.x_2 = self.x[1::2]
            self.x_1_ee = self.x_1[-1]
            self.x_2_ee = self.x_2[-1]
            self.x_ee = np.array([self.x_1_ee, self.x_2_ee])

    def __init__(self, arg):
    
            ## Model Parameters (Calculated)
            # Discrete length
            deltaL = RodLength / (N-1)
            # Radius of spheres
            R = np.zeros((N, 1))
            R[:] = deltaL/10.0

            all_q, all_u, timeArray = run_simulation(N, dt, totalTime, RodLength, deltaL, R, g, EI, EA, damp, m)
