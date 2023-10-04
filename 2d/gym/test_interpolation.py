# from scipy.interpolate import *
import numpy as np 
import matplotlib.pyplot as plt 

def quintic_func(q0, qf, T, qd0=0, qdf=0):
    X = [
        [ 0.0,          0.0,         0.0,        0.0,     0.0,  1.0],
        [ T**5,         T**4,        T**3,       T**2,    T,    1.0],
        [ 0.0,          0.0,         0.0,        0.0,     1.0,  0.0],
        [ 5.0 * T**4,   4.0 * T**3,  3.0 * T**2, 2.0 * T, 1.0,  0.0],
        [ 0.0,          0.0,         0.0,        2.0,     0.0,  0.0],
        [20.0 * T**3,  12.0 * T**2,  6.0 * T,    2.0,     0.0,  0.0],
    ]
    # fmt: on
    coeffs, resid, rank, s = np.linalg.lstsq(
        X, np.r_[q0, qf, qd0, qdf, 0, 0], rcond=None
    )

    # coefficients of derivatives
    coeffs_d = coeffs[0:5] * np.arange(5, 0, -1)
    coeffs_dd = coeffs_d[0:4] * np.arange(4, 0, -1)

    return lambda x: (
        np.polyval(coeffs, x),
        np.polyval(coeffs_d, x),
        np.polyval(coeffs_dd, x),
    )

def quintic(q0, qf, t, qd0=0, qdf=0):
    tf = max(t)

    polyfunc = quintic_func(q0, qf, tf, qd0, qdf)

    # evaluate the polynomials
    traj = polyfunc(t)
    p = traj[0]
    pd = traj[1]
    pdd = traj[2]

    return traj


x = np.array([0, 0.05, 0.5-0.05, 0.5])
y = np.array([0, 0, 1, 1])
x_interp = np.linspace(0, 0.5, 100)

traj = quintic(0, 1, np.linspace(0, 0.5, 100))

for i in range(3):  
    plt.figure()
    # f = PchipInterpolator(x, y, axis = 1).derivative(i)
    # f = CubicSpline(x, y, bc_type='clamped').derivative(i)
    # x_interp = np.linspace(0, 0.5, 100)

    # y_interp = f(x_interp)
    plt.plot(x_interp, traj[i])

plt.show()