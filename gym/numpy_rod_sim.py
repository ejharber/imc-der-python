# import cupy as np 
import numpy as np
from numba import jit

def run_simulation(x0, u0, N, dt, dL0, g, damp, m1, traj):
    # Initial DOF vector
    # Initial Position
    q0 = np.array([x0]).T

    # Initial velocity
    u0 = np.array([u0]).T

    # try:
    return _run_simulation(q0, u0, N, dt, dL0, g, damp, m1, traj)
    # except:
        # return [], [], [], False

@jit(cache=True, nopython=True)
def _run_simulation(q0, u0, N, dt, dL0, g, damp, m1, traj):
    # non-Homogenous Lengths
    dL = np.zeros((N-1))
    dL[0] = dL0

    ## Mass matrix
    masses = np.ones((2 * N, 1))
    masses[:] = m1 # mass of base 
    M = np.diag(masses.flatten())

    ## Gravity or weight vector
    W = masses * g
    W[::2] = 0.0

    # Tolerance
    tol = 1e-4 # division of rod length puts units in N

    q_save = np.zeros((q0.shape[0], traj.shape[1]))
    u_save = np.zeros((q0.shape[0], traj.shape[1]))
    f_save = np.zeros((q0.shape[0], traj.shape[1]))

    for c in range(traj.shape[1]):  # current time step is t_k+1 bc we start at a time of dt

        q = np.copy(q0)  # initial guess for newton raphson is the last dof vector
        q[:2, 0] = traj[:2, c]
        q[2, 0] = traj[0, c] + np.cos(traj[2, c]) * dL[0]
        q[3, 0] = traj[1, c] + np.sin(traj[2, c]) * dL[0]

        # Inertia
        f = M / dt @ ((q - q0) / dt - u0)  # inside() is new - old velocity, @ mKsns matrix multiplication
        J = M / dt ** 2.0

        # # Weight
        f = f - W

        # # Viscous force
        f = f - damp * (q - q0) / dt
        J = J - damp / dt

        # save values
        f_save[:, c] = np.copy(f[:, 0])
        q_save[:, c] = q[:,0]
        u_save[:, c] = u0[:,0]

        # update for next time step
        u0 = (q - q0) / dt  # New velocity becomes old velocity for next iter
        q0 = q  # Old position

    return q_save, u_save, f_save, True
