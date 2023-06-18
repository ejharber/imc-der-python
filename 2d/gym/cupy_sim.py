import cupy as cp 
import numpy as np
from numba import jit
from numba import cuda

def run_simulation(x0, u0, N, dt, RodLength, deltaL, R, g, EI, EA, damp, m, traj_u = np.zeros((3, 1))):
     # load into gpu


    x0 = cp.asarray(x0)
    u0 = cp.asarray(u0)
    # # N = cp.asarray(N)
    # # dt = cp.asarray(dt)
    # # RodLength = cp.asarray(RodLength)
    # # deltaL = cp.asarray(deltaL)
    # # R = cp.asarray(R)
    # # g = cp.asarray(g)
    # # EI = cp.asarray(EI)
    # # EA = cp.asarray(EA)
    # # damp = cp.asarray(damp)
    # # m = cp.asarray(m)
    traj_u = cp.asarray(traj_u)


    return _run_simulation(x0, u0, N, dt, RodLength, deltaL, R, g, EI, EA, damp, m, traj_u)

@cuda.jit
def _run_simulation(x0, u0, N, dt, RodLength, deltaL, R, g, EI, EA, damp, m, traj_u):

    ## Mass matrix
    # print(N)
    masses = cp.ones((2 * N, 1))
    masses = masses * m
    M = cp.diag(masses.flatten())

    ## Gravity or weight vector
    W = -masses * g
    W[::2] = 0

    # Initial DOF vector
    # Initial Position
    q0 = cp.reshape(x0, (-1, 1))

    # Initial velocity
    u0 = cp.reshape(u0, (-1, 1))

    # indeces for non - constrained motion
    free_bot = 4
    free_top = 2*N

    # Tolerance
    tol = 1e3 # division of rod length puts units in N

    q = 0

    for c in range(traj_u.shape[1]):  # current time step is t_k+1 bc we start at a time of dt

        q = cp.copy(q0)  # initial guess for newton raphson is the last dof vector

        q_ = q[:2] - q[2:4]
        rot = cp.array([[0, -1], [1, 0]])
        u_ = cp.atleast_2d(traj_u[:2, c]).T + traj_u[2, c] * rot @ q_ / cp.linalg.norm(q_) * deltaL

        q[:2, 0] += traj_u[:2, c]*dt
        q[2:4, 0] += u_[:,0]*dt

        q_free = q[free_bot:free_top]

        # Newton Raphson
        err = 10 * tol
        while err > tol:

            # Inertia
            # use @ for matrix multiplication else it's element wise
            f = M / dt @ ((q - q0) / dt - u0)  # inside() is new - old velocity, @ means matrix multiplication
            J = M / dt ** 2.0

            # Elastic forces
            k = 0  # do the first node or edge outside the loop
            # Linear spring between nodes k and k + 1
            indeces = [2*k, 2*k+1, 2*k+2, 2*k+3]
            xk = q[indeces[0]].item()  # xk
            yk = q[indeces[1]].item()  # yk
            xkp1 = q[indeces[2]].item()  # xk plus 1
            ykp1 = q[indeces[3]].item()
            dFs = grad_es(xk, yk, xkp1, ykp1, deltaL, EA)
            dJs = hess_es(xk, yk, xkp1, ykp1, deltaL, EA)
            indeces = cp.array(indeces)
            f[indeces] = f[indeces] + dFs  # check here if not working!!!!!!!!!!!!!!!!!!!!!!!1
            J[indeces[0]:indeces[3] + 1, indeces[0]:indeces[3] + 1] = J[indeces[0]:indeces[3] + 1,
                                                                      indeces[0]:indeces[3] + 1] + dJs
            for k in range(1, N-1):
                indeces = [2*k-2, 2*k-1, 2*k, 2*k+1, 2*k+2, 2*k+3]
                xkm1 = q[indeces[0]].item()  # xk minus 1
                ykm1 = q[indeces[1]].item()
                xk = q[indeces[2]].item()
                yk = q[indeces[3]].item()
                xkp1 = q[indeces[4]].item()
                ykp1 = q[indeces[5]].item()
                curvature0 = 0.0  # because we are using a straight beam

                indeces = cp.array(indeces)
                # linear spring between nodes k and k + 1
                dFs = grad_es(xk, yk, xkp1, ykp1, deltaL, EA)
                dJs = hess_es(xk, yk, xkp1, ykp1, deltaL, EA)
                f[indeces[2:]] = f[indeces[2:]] + dFs  # check here if not working!!!!!!!!!!!!!!!!!!!!!!!1
                J[indeces[2]:indeces[5] + 1, indeces[2]:indeces[5] + 1] = J[indeces[2]:indeces[5] + 1,
                                                                          indeces[2]:indeces[5] + 1] + dJs

                # Bending spring between nodes k-1, k, and k+1 located at node 2
                dFb = grad_eb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, deltaL, EI)
                dJb = hess_eb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, deltaL, EI)
                f[indeces] = f[indeces] + dFb  # and here !!!!!!!!!!!!!!!!!!!!!
                J[indeces[0]:indeces[5] + 1, indeces[0]:indeces[5] + 1] = J[indeces[0]:indeces[5] + 1,
                                                                          indeces[0]:indeces[5] + 1] + dJb

            # Weight
            f = f - W

            # Viscous force
            f = f + damp * (q - q0) / dt;
            J = J + damp / dt;

            # Update DOF for only the free ones
            f_free = cp.copy(f[free_bot:free_top])
            J_free = cp.copy(J[free_bot:free_top, free_bot:free_top])
            q_free -= cp.linalg.solve(J_free, f_free)
            q[free_bot:free_top] = q_free
            err = cp.sum(cp.absolute(f_free))  # error is sum of forces bc we want f=0

        # update
        u0 = (q - q0) / dt  # New velocity becomes old velocity for next iter
        q0 = q  # Old position

    q0 = cp.asnumpy(q0)
    u0 = cp.asnumpy(u0)

    return q0, u0, True

@cuda.jit
def cross_mat(a):
    """
    cross-matrix for derivative calculation
    :param a: numpy column or row array of size 3
    :return: 3x3 matrix w semi diagonal symmetry
    """
    c = [[0, -a[2], a[1]],
         [a[2], 0, -a[0]],
         [a[1], a[0], 0]]
    return cp.array(c)


@cuda.jit
def grad_eb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    This function returns the derivative of bending energy E_k^b with respect
    to x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    :param xkm1: this and following 5 must be pure number not nested arrays i.e xkm1 shouldn't equal ndarray[1.0]
    :param ykm1:
    :param xk:
    :param yk:
    :param xkp1:
    :param ykp1:
    :param curvature0: the "discrete" natural curvature [dimensionless] at node (xk, yk).
    :param l_k: the voronoi length of node (xk, yk).
    :param EI: the bending stiffness.
    :return: force
    """
    node0 = [xkm1, ykm1, 0.]
    node0 = cp.array(node0)
    node1 = [xk, yk, 0.]
    node1 = cp.array(node1)
    node2 = [xkp1, ykp1, 0.]
    node2 = cp.array(node2)
    # m1e,
    m2e = [0, 0, 1]
    m2e = cp.array(m2e, dtype="float64")
    # m1f,
    m2f = [0, 0, 1]
    m2f = cp.array(m2f, dtype="float64")
    kappaBar = curvature0

    # # Computation of gradient of the two curvatures
    gradKappa = cp.zeros((6, 1), dtype="float64")

    ee = node1 - node0
    ef = node2 - node1

    norm_e = cp.linalg.norm(ee)
    norm_f = cp.linalg.norm(ef)

    te = ee / norm_e
    tf = ef / norm_f

    # # Curvature binormal
    kb = 2.0 * cp.cross(te.flatten(), tf.flatten()) / (1.0 + cp.dot(te.flatten(), tf.flatten()))

    chi = 1.0 + cp.dot(te.flatten(), tf.flatten())
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # # Curvatures (check indexing in this section if any problems)
    kappa1 = kb[2]  # 0.5 * dot(kb, m2e + m2f); # CHECKED

    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + cp.cross(tf.flatten(), tilde_d2.flatten()))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - cp.cross(te.flatten(), tilde_d2.flatten()))

    gradKappa[0:2, 0] = -Dkappa1De[0:2]
    gradKappa[2:4, 0] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6, 0] = Dkappa1Df[0:2]

    # # Gradient of Eb
    dkappa = kappa1 - kappaBar
    dF = gradKappa * EI * dkappa / l_k

    return dF


@cuda.jit
def hess_eb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    This function returns the derivative of bending energy E_k^b with respect
    to x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    :param xkm1:
    :param ykm1:
    :param xk:
    :param yk:
    :param xkp1:
    :param ykp1:
    :param curvature0: the "discrete" natural curvature [dimensionless] at node (xk, yk).
    :param l_k: the voronoi length of node (xk, yk).
    :param EI: the bending stiffness.
    :return: jacobian
    """
    node0 = [xkm1, ykm1, 0]
    node0 = cp.array(node0)
    node1 = [xk, yk, 0]
    node1 = cp.array(node1)
    node2 = [xkp1, ykp1, 0]
    node2 = cp.array(node2)
    # m1e,
    m2e = [0, 0, 1]
    m2e = cp.array(m2e, dtype="float64")
    # m1f,
    m2f = [0, 0, 1]
    m2f = cp.array(m2f, dtype="float64")
    kappaBar = curvature0

    # # Computation of gradient of the two curvatures
    gradKappa = cp.zeros((6, 1), dtype="float64")

    ee = node1 - node0
    ef = node2 - node1

    norm_e = cp.linalg.norm(ee)
    norm_f = cp.linalg.norm(ef)

    te = ee / norm_e
    tf = ef / norm_f

    # # Curvature binormal
    kb = 2.0 * cp.cross(te.flatten(), tf.flatten()) / (1.0 + cp.dot(te.flatten(), tf.flatten()))

    chi = 1.0 + cp.dot(te.flatten(), tf.flatten())
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # # Curvatures (check indexing in this section if any problems)
    kappa1 = kb[2]  # 0.5 * dot(kb, m2e + m2f); # CHECKED

    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + cp.cross(tf.flatten(), tilde_d2.flatten()))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - cp.cross(te.flatten(), tilde_d2.flatten()))

    gradKappa[0:2, 0] = -Dkappa1De[0:2]
    gradKappa[2:4, 0] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6, 0] = Dkappa1Df[0:2]

    # # Computation of hessian of the two curvatures
    DDkappa1 = cp.zeros((6, 6))
    # DDkappa2 = zeros(11, 11)

    norm2_e = norm_e ** 2
    norm2_f = norm_f ** 2

    tt_o_tt = cp.transpose(tilde_t) * tilde_t  # must be 3x3. tilde_t is 1x3
    # tt_o_tt = cp.outer(tilde_t, tilde_t)  # alternative bc I think it's just outer product, try for following too
    tmp = cp.cross(tf.flatten(), tilde_d2.flatten())
    tf_c_d2t_o_tt = cp.transpose(tmp) * tilde_t  # must be 3x3
    tt_o_tf_c_d2t = cp.transpose(tf_c_d2t_o_tt)  # must be 3x3
    kb_o_d2e = cp.transpose(kb) * m2e  # must be 3x3
    d2e_o_kb = cp.transpose(kb_o_d2e)  # must be 3x3

    Id3 = cp.identity(3)
    te_transpose = cp.transpose(te)
    D2kappa1De2 = 1.0 / norm2_e * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tt_o_tf_c_d2t) - kappa1 / (chi * norm2_e) \
                  * (Id3 - te_transpose * te) + 1.0 / (4.0 * norm2_e) * (kb_o_d2e + d2e_o_kb)

    tmp = cp.cross(te.flatten(), tilde_d2.flatten())
    te_c_d2t_o_tt = cp.transpose(tmp) * tilde_t
    tt_o_te_c_d2t = cp.transpose(te_c_d2t_o_tt)
    kb_o_d2f = cp.transpose(kb) * m2f
    d2f_o_kb = kb_o_d2f

    tf_transpose = cp.transpose(tf)
    D2kappa1Df2 = 1.0 / norm2_f * (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + tt_o_te_c_d2t) - kappa1 / (chi * norm2_f) \
                  * (Id3 - tf_transpose * tf) + 1.0 / (4.0 * norm2_f) * (kb_o_d2f + d2f_o_kb)

    D2kappa1DeDf = -kappa1/(chi * norm_e * norm_f) * (Id3 + te_transpose*tf) + 1.0 / (norm_e*norm_f) \
                   * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + tt_o_te_c_d2t - cross_mat(tilde_d2))

    D2kappa1DfDe = cp.transpose(D2kappa1DeDf)

    # Curvature terms (check indexing)
    DDkappa1[0: 2, 0: 2] = D2kappa1De2[0: 2, 0: 2]
    DDkappa1[0: 2, 2: 4] = - D2kappa1De2[0: 2, 0: 2] + D2kappa1DeDf[0: 2, 0: 2]
    DDkappa1[0: 2, 4: 6] = - D2kappa1DeDf[0: 2, 0: 2]
    DDkappa1[2: 4, 0: 2] = - D2kappa1De2[0: 2, 0: 2] + D2kappa1DfDe[0: 2, 0: 2]
    DDkappa1[2: 4, 2: 4] = D2kappa1De2[0: 2, 0: 2] - D2kappa1DeDf[0: 2, 0: 2] - D2kappa1DfDe[0: 2, 0: 2] \
                           + D2kappa1Df2[0: 2, 0: 2]
    DDkappa1[2: 4, 4: 6] = D2kappa1DeDf[0: 2, 0: 2] - D2kappa1Df2[0: 2, 0: 2]
    DDkappa1[4: 6, 0: 2] = - D2kappa1DfDe[0: 2, 0: 2]
    DDkappa1[4: 6, 2: 4] = D2kappa1DfDe[0: 2, 0: 2] - D2kappa1Df2[0: 2, 0: 2]
    DDkappa1[4: 6, 4: 6] = D2kappa1Df2[0: 2, 0: 2]

    # # Hessian of Eb
    dkappa = kappa1 - kappaBar
    dJ = 1.0 / l_k * EI * gradKappa * cp.transpose(gradKappa)
    temp = 1.0 / l_k * dkappa * EI
    dJ = dJ + temp * DDkappa1

    return dJ


@cuda.jit
def grad_es(xk, yk, xkp1, ykp1, l_k, EA):
    """
    This function returns the derivative of stretching energy E_k^s with
    respect to x_{k-1}, y_{k-1}, x_k, and y_k
    :param xk:
    :param yk:
    :param xkp1:
    :param ykp1:
    :param l_k:
    :param EA:
    :return:
    """

    F = cp.zeros((4, 1))

    F[0] = -(0.1e1 - cp.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (-0.2e1 * xkp1 + 0.2e1 * xk)
    F[1] = -(0.1e1 - cp.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (-0.2e1 * ykp1 + 0.2e1 * yk)
    F[2] = -(0.1e1 - cp.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (0.2e1 * xkp1 - 0.2e1 * xk)
    F[3] = -(0.1e1 - cp.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (0.2e1 * ykp1 - 0.2e1 * yk)

    F = 0.5 * EA * l_k * F

    return F


@cuda.jit
def hess_es(xk, yk, xkp1, ykp1, l_k, EA):
    """
    This function returns the 4x4 hessian of the stretching energy E_k^s with
    respect to x_k, y_k, x_{k+1}, and y_{k+1}.
    :param xk:
    :param yk:
    :param xkp1:
    :param ykp1:
    :param l_k:
    :param EA:
    :return:
    """
    J11 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 + (
                0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (
                      (-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (
                      0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J12 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (
                0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (
                      -2 * ykp1 + 2 * yk) / 0.2e1
    J13 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (
                0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (
                      2 * xkp1 - 2 * xk) / 0.2e1 + 0.2e1 * (0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J14 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (
                0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (
                      2 * ykp1 - 2 * yk) / 0.2e1
    J22 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 + (
                0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (
                      (-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (
                      0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J23 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (
                0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (
                      2 * xkp1 - 2 * xk) / 0.2e1
    J24 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (
                0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (
                      2 * ykp1 - 2 * yk) / 0.2e1 + 0.2e1 * (0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J33 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) ** 2) / 0.2e1 + (
                0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (
                      (2 * xkp1 - 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (
                      0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J34 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (2 * xkp1 - 2 * xk)) / 0.2e1 + (
                0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (2 * xkp1 - 2 * xk) * (
                      2 * ykp1 - 2 * yk) / 0.2e1
    J44 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) ** 2) / 0.2e1 + (
                0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (
                      (2 * ykp1 - 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (
                      0.1e1 - cp.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (
                      ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k

    J = [
        [J11, J12, J13, J14],
        [J12, J22, J23, J24],
        [J13, J23, J33, J34],
        [J14, J24, J34, J44]
    ]

    J = cp.array(J)

    J = cp.reshape(J, (4, 4))

    J = 0.5 * EA * l_k * J

    return J