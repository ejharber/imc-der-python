import numpy as np 

def run_simulation(N, dt, totalTime, RodLength, deltaL, R, g, EI, EA, damp, m):

    ## Geometry
    nodes = np.zeros((N, 2))
    for c in range(N):
        nodes[c, 0] = c * deltaL

    ## Mass matrix
    masses = np.ones((2 * N, 1))
    masses = masses * m
    M = np.diag(masses.flatten())

    ## Gravity or weight vector
    W = -masses * g
    W[::2] = 0

    # Initial DOF vector
    # Initial Position
    q0 = np.reshape(nodes, (-1, 1))

    # Initial velocity
    u0 = np.zeros((2 * N, 1))  # zero initial velocity

    # indeces for non - constrained motion
    free_bot = 2
    free_top = 2*N

    # # Time marching scheme
    # Number of time steps
    Nsteps = int(np.round(totalTime / dt) + 1)  # add 1 to include time of 0

    # Tolerance
    tol = EI / RodLength ** 2 * 1e-3  # division of rod length puts units in N

    q = 0

    for c in range(1, Nsteps):  # current time step is t_k+1 bc we start at a time of dt

        # print('Old Time = {:.2}\n, New Time = {:.2}\n'.format((c - 1) * dt, c * dt))

        q = np.copy(q0)  # initial guess for newton raphson is the last dof vector

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
            dFs = funcs.grad_es(xk, yk, xkp1, ykp1, deltaL, EA)
            dJs = funcs.hess_es(xk, yk, xkp1, ykp1, deltaL, EA)
            indeces = np.array(indeces)
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

                indeces = np.array(indeces)
                # linear spring between nodes k and k + 1
                dFs = funcs.grad_es(xk, yk, xkp1, ykp1, deltaL, EA)
                dJs = funcs.hess_es(xk, yk, xkp1, ykp1, deltaL, EA)
                f[indeces[2:]] = f[indeces[2:]] + dFs  # check here if not working!!!!!!!!!!!!!!!!!!!!!!!1
                J[indeces[2]:indeces[5] + 1, indeces[2]:indeces[5] + 1] = J[indeces[2]:indeces[5] + 1,
                                                                          indeces[2]:indeces[5] + 1] + dJs

            # Weight
            f = f - W

            # Viscous force
            f = f + damp * (q - q0) / dt;
            J = J + damp / dt;

            # Update DOF for only the free ones
            f_free = np.copy(f[free_bot:free_top])
            J_free = np.copy(J[free_bot:free_top, free_bot:free_top])
            q_free -= np.linalg.solve(J_free, f_free)
            q[free_bot:free_top] = q_free
            err = np.sum(np.absolute(f_free))  # error is sum of forces bc we want f=0

        # update
        u0 = (q - q0) / dt  # New velocity becomes old velocity for next iter
        q0 = q  # Old position

    return all_q, all_u, timeArray


def cross_mat(a):
    """
    cross-matrix for derivative calculation
    :param a: numpy column or row array of size 3
    :return: 3x3 matrix w semi diagonal symmetry
    """
    c = [[0, -a[2], a[1]],
         [a[2], 0, -a[0]],
         [a[1], a[0], 0]]
    return np.array(c)

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
    node0 = np.array(node0)
    node1 = [xk, yk, 0.]
    node1 = np.array(node1)
    node2 = [xkp1, ykp1, 0.]
    node2 = np.array(node2)
    # m1e,
    m2e = [0, 0, 1]
    m2e = np.array(m2e, dtype="float64")
    # m1f,
    m2f = [0, 0, 1]
    m2f = np.array(m2f, dtype="float64")
    kappaBar = curvature0

    # # Computation of gradient of the two curvatures
    gradKappa = np.zeros((6, 1), dtype="float64")

    ee = node1 - node0
    ef = node2 - node1

    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    te = ee / norm_e
    tf = ef / norm_f

    # # Curvature binormal
    kb = 2.0 * np.cross(te.flatten(), tf.flatten()) / (1.0 + np.dot(te.flatten(), tf.flatten()))

    chi = 1.0 + np.dot(te.flatten(), tf.flatten())
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # # Curvatures (check indexing in this section if any problems)
    kappa1 = kb[2]  # 0.5 * dot(kb, m2e + m2f); # CHECKED

    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf.flatten(), tilde_d2.flatten()))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te.flatten(), tilde_d2.flatten()))

    gradKappa[0:2, 0] = -Dkappa1De[0:2]
    gradKappa[2:4, 0] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6, 0] = Dkappa1Df[0:2]

    # # Gradient of Eb
    dkappa = kappa1 - kappaBar
    dF = gradKappa * EI * dkappa / l_k

    return dF

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
    node0 = np.array(node0)
    node1 = [xk, yk, 0]
    node1 = np.array(node1)
    node2 = [xkp1, ykp1, 0]
    node2 = np.array(node2)
    # m1e,
    m2e = [0, 0, 1]
    m2e = np.array(m2e, dtype="float64")
    # m1f,
    m2f = [0, 0, 1]
    m2f = np.array(m2f, dtype="float64")
    kappaBar = curvature0

    # # Computation of gradient of the two curvatures
    gradKappa = np.zeros((6, 1), dtype="float64")

    ee = node1 - node0
    ef = node2 - node1

    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    te = ee / norm_e
    tf = ef / norm_f

    # # Curvature binormal
    kb = 2.0 * np.cross(te.flatten(), tf.flatten()) / (1.0 + np.dot(te.flatten(), tf.flatten()))

    chi = 1.0 + np.dot(te.flatten(), tf.flatten())
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # # Curvatures (check indexing in this section if any problems)
    kappa1 = kb[2]  # 0.5 * dot(kb, m2e + m2f); # CHECKED

    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf.flatten(), tilde_d2.flatten()))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te.flatten(), tilde_d2.flatten()))

    gradKappa[0:2, 0] = -Dkappa1De[0:2]
    gradKappa[2:4, 0] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6, 0] = Dkappa1Df[0:2]

    # # Computation of hessian of the two curvatures
    DDkappa1 = np.zeros((6, 6))
    # DDkappa2 = zeros(11, 11)

    norm2_e = norm_e ** 2
    norm2_f = norm_f ** 2

    tt_o_tt = np.transpose(tilde_t) * tilde_t  # must be 3x3. tilde_t is 1x3
    # tt_o_tt = np.outer(tilde_t, tilde_t)  # alternative bc I think it's just outer product, try for following too
    tmp = np.cross(tf.flatten(), tilde_d2.flatten())
    tf_c_d2t_o_tt = np.transpose(tmp) * tilde_t  # must be 3x3
    tt_o_tf_c_d2t = np.transpose(tf_c_d2t_o_tt)  # must be 3x3
    kb_o_d2e = np.transpose(kb) * m2e  # must be 3x3
    d2e_o_kb = np.transpose(kb_o_d2e)  # must be 3x3

    Id3 = np.identity(3)
    te_transpose = np.transpose(te)
    D2kappa1De2 = 1.0 / norm2_e * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tt_o_tf_c_d2t) - kappa1 / (chi * norm2_e) \
                  * (Id3 - te_transpose * te) + 1.0 / (4.0 * norm2_e) * (kb_o_d2e + d2e_o_kb)

    tmp = np.cross(te.flatten(), tilde_d2.flatten())
    te_c_d2t_o_tt = np.transpose(tmp) * tilde_t
    tt_o_te_c_d2t = np.transpose(te_c_d2t_o_tt)
    kb_o_d2f = np.transpose(kb) * m2f
    d2f_o_kb = kb_o_d2f

    tf_transpose = np.transpose(tf)
    D2kappa1Df2 = 1.0 / norm2_f * (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + tt_o_te_c_d2t) - kappa1 / (chi * norm2_f) \
                  * (Id3 - tf_transpose * tf) + 1.0 / (4.0 * norm2_f) * (kb_o_d2f + d2f_o_kb)

    D2kappa1DeDf = -kappa1/(chi * norm_e * norm_f) * (Id3 + te_transpose*tf) + 1.0 / (norm_e*norm_f) \
                   * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + tt_o_te_c_d2t - cross_mat(tilde_d2))

    D2kappa1DfDe = np.transpose(D2kappa1DeDf)

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
    dJ = 1.0 / l_k * EI * gradKappa * np.transpose(gradKappa)
    temp = 1.0 / l_k * dkappa * EI
    dJ = dJ + temp * DDkappa1

    return dJ

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

    F = np.zeros((4, 1))

    F[0] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (-0.2e1 * xkp1 + 0.2e1 * xk)
    F[1] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (-0.2e1 * ykp1 + 0.2e1 * yk)
    F[2] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (0.2e1 * xkp1 - 0.2e1 * xk)
    F[3] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (
                -0.1e1 / 0.2e1) / l_k * (0.2e1 * ykp1 - 0.2e1 * yk)

    F = 0.5 * EA * l_k * F

    return F

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

    J = 0.5 * EA * l_k * J

    return J