# import cupy as np 
import numpy as np
from numba import jit

def run_simulation(x0, u0, N, dt, g, dL, Ks, damp, m, traj, axial_damp):
    # Initial DOF vector
    # Initial Position
    q0 = np.array([x0]).T

    # Initial velocity
    u0 = np.array([u0]).T

    # try:
    return _run_simulation(q0, u0, N, dt, g, dL, Ks, damp, m, traj, axial_damp)
    # except:
    return [], [], [], False

def _run_simulation(q0, u0, N, dt, g, dL_, Ks_, damp_, m_, traj, axial_damp):

    # non-Homoegneous Ks
    Ks = np.zeros((N-1))
    Ks[0] = Ks_

    # non-Homogenous Lengths
    dL = np.zeros((N-1))
    dL[0] = dL_

    # # non-Homogenous Axial Dampening
    # axial_damp = np.zeros((N-1))
    # axial_damp[0] = axial_damp_

    ## Mass matrix
    masses = np.ones((2 * N, 1))
    masses[:] = m_ # mass of base 
    M = np.diag(masses.flatten())

    ## Gravity or weight vector
    W = masses * g
    W[::2] = 0.0

    # indeces for non - constrained motion
    free_bot = 2
    free_top = 2*N

    # Tolerance
    tol = 1e-6 # division of rod length puts units in N

    q_save = np.zeros((q0.shape[0], traj.shape[1]))
    u_save = np.zeros((q0.shape[0], traj.shape[1]))
    f_save = np.zeros((q0.shape[0], traj.shape[1]))

    print(traj.shape[1])

    for c in range(traj.shape[1]):  # current time step is t_k+1 bc we start at a time of dt

        q = np.copy(q0)  # initial guess for newton raphson is the last dof vector
        q_free = q[free_bot:free_top]
        q[:2, 0] = traj[:2, c]

        # Newton Raphson
        err = 10 * tol

        iteration = 0
        max_iteration = 500_000

        while err > tol:

            iteration += 1

            if iteration >= max_iteration:
                print('max iter reached')
                return q_save, u_save, f_save, False

            # Inertia
            f = M / dt @ ((q - q0) / dt - u0)  # inside() is new - old velocity, @ mKsns matrix multiplication
            J = M / dt ** 2.0

            # # Weight
            f = f - W

            # # Viscous force
            f = f - damp_ * (q - q0) / dt
            J = J - damp_ / dt

            k = 0
            # Linear spring between nodes k and k + 1
            indeces = [2*k, 2*k+1, 2*k+2, 2*k+3]
            xk = q[indeces[0]].item()  # xk
            yk = q[indeces[1]].item()  # yk
            xkp1 = q[indeces[2]].item()  # xk plus 1
            ykp1 = q[indeces[3]].item()

            dFs = grad_es(xk, yk, xkp1, ykp1, dL[k], Ks[k])                
            dJs = hess_es(xk, yk, xkp1, ykp1, dL[k], Ks[k])
            indeces = np.array(indeces)
            f[indeces] = f[indeces] + dFs  # check here if not working!
            # print(f_save[indeces, c].shape, dFs.shape)
            J[indeces[0]:indeces[3] + 1, indeces[0]:indeces[3] + 1] = J[indeces[0]:indeces[3] + 1,
                                                                      indeces[0]:indeces[3] + 1] + dJs
            
            k = 0
            indeces = [2*k-2, 2*k-1, 2*k, 2*k+1, 2*k+2, 2*k+3]
            indeces = np.array(indeces)

            # xkm1 = q[indeces[0]].item()  # xk minus 1
            # ykm1 = q[indeces[1]].item()
            xk = q[indeces[2]].item()
            yk = q[indeces[3]].item()
            xkp1 = q[indeces[4]].item()
            ykp1 = q[indeces[5]].item()

            # xk0m1 = q0[indeces[0]].item()  # xk minus 1
            # yk0m1 = q0[indeces[1]].item()
            x0k = q0[indeces[2]].item()  # xk
            y0k = q0[indeces[3]].item()  # yk
            x0kp1 = q0[indeces[4]].item()  # xk
            y0kp1 = q0[indeces[5]].item()  # yk

            spring_vec = np.array([xkp1, ykp1]) - np.array([xk, yk])
            uk = np.array([xk - x0k, yk - y0k]) / dt
            ukp1 = np.array([xkp1 - x0kp1, ykp1 - y0kp1]) / dt

            u_vec = ukp1 - uk

            if np.dot(u_vec, spring_vec) < 0:
                spring_vec = - spring_vec 

            dFs = axial_damp * np.atleast_2d((np.dot(u_vec, spring_vec) / np.dot(spring_vec, spring_vec)) * spring_vec).T

            # Compute Jacobian for the viscous force term
            spring_norm_sq = np.dot(spring_vec, spring_vec)
            u_dot_spring = np.dot(u_vec, spring_vec)

            f[indeces[4:6]] = f[indeces[4:6]] + dFs
            
            # f_save[indeces[2:4], c] = f_save[indeces[2:4], c] + dFs[:2, 0]

            # Derivatives of dFs with respect to q (Jacobian)
            dFdq_kkp1 = axial_damp / dt * (np.outer(spring_vec, u_vec) / spring_norm_sq)

            # Add to Jacobian matrix for the indices of q corresponding to the forces
            for i in range(2):
                for j in range(2):
                    J[indeces[4 + i], indeces[4 + j]] += dFdq_kkp1[i, j]


            f_free = np.copy(f[free_bot:free_top])
            J_free = np.copy(J[free_bot:free_top, free_bot:free_top])
            q_free -= np.linalg.solve(J_free, f_free)
            q[free_bot:free_top] = q_free
            err = np.sum(np.absolute(f_free))  # error is sum of forces bc we want f=0


        # update
        u0 = (q - q0) / dt  # New velocity becomes old velocity for next iter
        q0 = q  # Old position

        f_save[:, c] = np.copy(f[:, 0])
        q_save[:, c] = q[:,0]
        u_save[:, c] = u0[:,0]
            
    return q_save, u_save, f_save, True


@jit(cache=True, nopython=True)
def grad_es(xk, yk, xkp1, ykp1, l_k, Ks):
    """
    This function returns the derivative of stretching energy E_k^s with
    respect to x_{k-1}, y_{k-1}, x_k, and y_k
    :param xk:
    :param yk:
    :param xkp1:
    :param ykp1:
    :param l_k:
    :param Ks:
    :return:
    """

    F = np.zeros((4, 1))

    F[0] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (-0.2e1 * xkp1 + 0.2e1 * xk)
    F[1] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (-0.2e1 * ykp1 + 0.2e1 * yk)
    F[2] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (0.2e1 * xkp1 - 0.2e1 * xk)
    F[3] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (0.2e1 * ykp1 - 0.2e1 * yk)

    F = 0.5 * Ks * l_k * F

    return F

@jit(cache=True, nopython=True)
def hess_es(xk, yk, xkp1, ykp1, l_k, Ks):
    """
    This function returns the 4x4 hessian of the stretching energy E_k^s with
    respect to x_k, y_k, x_{k+1}, and y_{k+1}.
    :param xk:
    :param yk:
    :param xkp1:
    :param ykp1:
    :param l_k:
    :param Ks:
    :return:
    """
    J11 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 + (
                0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (
                      (-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (
                      0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J12 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (
                0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (
                      -2 * ykp1 + 2 * yk) / 0.2e1
    J13 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (
                0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (
                      2 * xkp1 - 2 * xk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J14 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (
                0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (
                      2 * ykp1 - 2 * yk) / 0.2e1
    J22 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 + (
                0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (
                      (-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (
                      0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J23 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (
                0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (
                      2 * xkp1 - 2 * xk) / 0.2e1
    J24 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (
                0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (
                      2 * ykp1 - 2 * yk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J33 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) ** 2) / 0.2e1 + (
                0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (
                      (2 * xkp1 - 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (
                      0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J34 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (2 * xkp1 - 2 * xk)) / 0.2e1 + (
                0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (2 * xkp1 - 2 * xk) * (
                      2 * ykp1 - 2 * yk) / 0.2e1
    J44 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) ** 2) / 0.2e1 + (
                0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (
                      (2 * ykp1 - 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (
                      0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k

    J = [
        [J11, J12, J13, J14],
        [J12, J22, J23, J24],
        [J13, J23, J33, J34],
        [J14, J24, J34, J44]
    ]

    J = np.array(J)

    J = np.reshape(J, (4, 4))

    J = 0.5 * Ks * l_k * J

    return J
