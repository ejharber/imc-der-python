# import cupy as np 
import numpy as np
from numba import jit

def run_simulation(x0, u0, N, dt, dL_stick, dL_ati, dL_rope, Kb, Kb_connector, Ks, Ks_connector, damp, rope_damp, ati_damp, m_holder, m_rope, m_tip, traj):
    # Initial DOF vector
    # Initial Position
    q0 = np.array([x0]).T

    # Initial velocity
    u0 = np.array([u0]).T

    try:
        return _run_simulation(q0, u0, N, dt, dL_stick, dL_ati, dL_rope, Kb, Kb_connector, Ks, Ks_connector, damp, rope_damp, ati_damp, m_holder, m_rope, m_tip, traj)
    except:
        print('error')
        return [], [], [],  False

@jit(cache=True, nopython=True)
def _run_simulation(q0, u0, N, dt, dL_stick, dL_ati, dL_rope, Kb_rope, Kb_connector, Ks_rope, Ks_connector, damp, rope_damp, ati_damp, m_holder, m_rope, m_tip, traj):
    # Set up stiffness and damping coefficients
    Kb = np.zeros((N-1))
    Kb[:3] = Kb_connector
    Kb[3:] = Kb_rope

    Ks = np.zeros((N-1))
    Ks[:3] = Ks_connector
    Ks[3:] = Ks_rope

    dL = np.zeros((N-1))
    dL[0] = dL_stick
    dL[1:3] = dL_ati
    dL[3:] = dL_rope

    axial_damp = np.zeros((N-1))
    axial_damp[:3] = ati_damp
    axial_damp[3:] = rope_damp

    masses = np.ones((2 * N, 1))
    masses[:6, :] = m_holder  # Mass of holder
    masses[6:-2, :] = m_rope  # Mass of rope
    masses[-2:, :] = m_tip    # Mass of tip
    M = np.diag(masses.flatten())

    W = masses * 9.81
    W[::2] = 0.0  # Apply gravity only in the y-direction

    free_bot = 4
    free_top = 2 * N
    tol = 1e-2

    q_save = np.zeros((q0.shape[0], traj.shape[1]))
    u_save = np.zeros((q0.shape[0], traj.shape[1]))
    f_save = np.zeros((q0.shape[0], traj.shape[1]))  # Stores internal forces

    for c in range(traj.shape[1]):
        q = np.copy(q0)
        q[:2, 0] = traj[:2, c]
        q[2, 0] = traj[0, c] + np.cos(traj[2, c]) * dL[0]
        q[3, 0] = traj[1, c] + np.sin(traj[2, c]) * dL[0]
        q_free = q[free_bot:free_top]

        err = 10 * tol
        iteration = 0
        max_iteration = 50_000

        while err > tol:
            iteration += 1
            if iteration >= max_iteration or err > 1e3 or np.any(np.isnan(q)):
                print("max iter reached", c, err)
                return q_save, u_save, f_save, False

            f_save[:, c] = f_save[:, c] * 0

            f = M / dt @ ((q - q0) / dt - u0)  # Inertia
            J = M / dt ** 2.0
            f += W  # Weight
            f += damp * (q - q0) / dt  # Viscous damping
            J += damp / dt

            for k in range(N-1):
                indices = np.array([2*k, 2*k+1, 2*k+2, 2*k+3])
                xk, yk = q[indices[0]].item(), q[indices[1]].item()
                xkp1, ykp1 = q[indices[2]].item(), q[indices[3]].item()
                uk, vk = u0[indices[0]].item(), u0[indices[1]].item()
                ukp1, vkp1 = u0[indices[2]].item(), u0[indices[3]].item()

                # Stretching forces
                dFs = grad_es(xk, yk, xkp1, ykp1, dL[k], Ks[k])
                dJs = hess_es(xk, yk, xkp1, ykp1, dL[k], Ks[k])
                f[indices] += dFs
                J[indices[0]:indices[3] + 1, indices[0]:indices[3] + 1] += dJs

                # Update f_save for stretching forces
                f_save[indices[:2], c] -= dFs[2:, 0]
                f_save[indices[2:], c] += dFs[:2, 0]

                # # Axial damping (extrinsic model)
                # dx, dy = xkp1 - xk, ykp1 - yk
                # length = np.sqrt(dx**2 + dy**2)
                # nx, ny = dx / length, dy / length  # Unit vector along the axis
                # rel_vel_x, rel_vel_y = ukp1 - uk, vkp1 - vk
                # rel_vel = rel_vel_x * nx + rel_vel_y * ny  # Relative velocity along the axis
                # damping_force = -axial_damp[k] * rel_vel

                # dFad = np.array([[damping_force * nx, damping_force * ny, -damping_force * nx, -damping_force * ny]])
                # f[indices] += dFad

                # # Update f_save for stretching forces
                # f_save[indices[:2], c] -= dFad[2:, 0]
                # f_save[indices[2:], c] += dFad[:2, 0]

                # # Add damping contribution to Jacobian
                # dJ_damp = axial_damp[k] * np.array([
                #     [nx*nx, nx*ny, -nx*nx, -nx*ny],
                #     [nx*ny, ny*ny, -nx*ny, -ny*ny],
                #     [-nx*nx, -nx*ny, nx*nx, nx*ny],
                #     [-nx*ny, -ny*ny, nx*ny, ny*ny]
                # ])
                # J[indices[0]:indices[3] + 1, indices[0]:indices[3] + 1] += dJ_damp

            # Bending forces (for k >= 1)
            for k in range(1, N-1):
                indices = np.array([2*k-2, 2*k-1, 2*k, 2*k+1, 2*k+2, 2*k+3])
                xkm1, ykm1 = q[indices[0]].item(), q[indices[1]].item()
                xk, yk = q[indices[2]].item(), q[indices[3]].item()
                xkp1, ykp1 = q[indices[4]].item(), q[indices[5]].item()

                # Compute bending forces and Jacobian
                curvature0 = 0.0  # Assume a straight rod initially
                dFb = grad_eb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, dL[k], Kb[k])
                dJb = hess_eb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, dL[k], Kb[k])
                f[indices] += dFb
                J[indices[0]:indices[5] + 1, indices[0]:indices[5] + 1] += dJb

                # Update f_save for bending forces
                # f_save[indices, c] += dFb[:, 0]

            # Update positions using Newton-Raphson
            f_free = f[free_bot:free_top]
            J_free = J[free_bot:free_top, free_bot:free_top]
            q_free -= np.linalg.solve(J_free, f_free)
            q[free_bot:free_top] = q_free
            err = np.sum(np.absolute(f_free))  # error is sum of forces bc we want f=0

        u0 = (q - q0) / dt
        q0 = q
        q_save[:, c] = q[:, 0]
        u_save[:, c] = u0[:, 0]

    return q_save, u_save, f_save, True

@jit(cache=True, nopython=True)
def cross_mat(a):
    """
    cross-matrix for derivative calculation
    :param a: numpy column or row array of size 3
    :return: 3x3 matrix w semi diagonal symmetry
    """
    c = [[0, -a[2], a[1]],
         [a[2], 0, -a[0]],
         [-a[1], a[0], 0]]
    return np.array(c)


@jit(cache=True, nopython=True)
def grad_eb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, Kb):
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
    :param Kb: the bending stiffness.
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
    chi = max(1e-4, chi)

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
    dF = gradKappa * Kb * dkappa / l_k

    return dF


@jit(cache=True, nopython=True)
def hess_eb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, Kb):
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
    :param Kb: the bending stiffness.
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
    chi = max(1e-4, chi)

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
    dJ = 1.0 / l_k * Kb * gradKappa * np.transpose(gradKappa)
    temp = 1.0 / l_k * dkappa * Kb
    dJ = dJ + temp * DDkappa1

    return dJ


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