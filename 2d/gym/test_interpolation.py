# from scipy.interpolate import *
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set() # Setting seaborn as default style even if use only matplotlib

import matplotlib.font_manager

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 16

# able'), FontEntry(fname='/usr/share/fonts/truetype/tlwg/Garuda.ttf', name='Garuda', style='normal', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf', name='Nimbus Mono PS', style='normal', variant='normal', weight=700, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/opentype/urw-base35/C059-Italic.otf', name='C059', style='italic', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/msttcorefonts/Verdana_Bold.ttf', name='Verdana', style='normal', variant='normal', weight=700, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/tlwg/TlwgTypist-Oblique.ttf', name='Tlwg Typist', style='oblique', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/tlwg/TlwgMono-BoldOblique.ttf', name='Tlwg Mono', style='oblique', variant='normal', weight=700, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', name='DejaVu Sans Mono', style='normal', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Oblique.otf', name='Nimbus Sans Narrow', style='oblique', variant='normal', weight=400, stretch='condensed', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf', name='Arial', style='italic', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-BoldItalic.ttf', name='DejaVu Serif', style='italic', variant='normal', weight=700, stretch='condensed', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/teluguvijayam/TimmanaRegular.ttf', name='Timmana', style='normal', variant='normal', weight=900, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/opentype/urw-base35/NimbusRoman-BoldItalic.otf', name='Nimbus Roman', style='italic', variant='normal', weight=700, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', name='Liberation Sans', style='normal', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/lyx/stmary10.ttf', name='stmary10', style='normal', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/tlwg/Laksaman-Italic.ttf', name='Laksaman', style='italic', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-BoldOblique.ttf', name='DejaVu Sans', style='oblique', variant='normal', weight=700, stretch='condensed', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/msttcorefonts/courbi.ttf', name='Courier New', style='italic', variant='normal', weight=700, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/teluguvijayam/Gurajada.ttf', name='Gurajada', style='normal', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Regular.ttf', name='Yrsa', style='normal', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/msttcorefonts/Georgia_Italic.ttf', name='Georgia', style='italic', variant='normal', weight=400, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/tlwg/Kinnari-Bold.ttf', name='Kinnari', style='normal', variant='normal', weight=700, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Italic.ttf', name='Liberation Sans Narrow', style='italic', variant='normal', weight=400, stretch='condensed', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/msttcorefonts/timesbi.ttf', name='Times New Roman', style='italic', variant='normal', weight=700, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/fonts-beng-extra/LikhanNormal.ttf', name='Likhan', style='normal', variant='normal', weight=500, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/truetype/liberation2/LiberationSans-BoldItalic.ttf', name='Liberation Sans', style='italic', variant='normal', weight=700, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/opentype/malayalam/Manjari-Thin.otf', name='Manjari', style='normal', variant='normal', weight=100, stretch='normal', size='scalable'), FontEntry(fname='/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Italic.otf', name='Nimbus Mono PS', style='italic', variant='normal', weight=400, stretch='normal', size='scalable')]

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

t_interp = np.linspace(0, 0.5, 100)
traj_x = quintic(0, 0.1, t_interp)
traj_y = quintic(0, -0.1, t_interp)
traj_theta = quintic(0, np.pi/4, t_interp)

plt.figure("Position")
plt.title("Position", fontsize = 20, fontweight='bold')
plt.plot(t_interp, traj_x[0], label='x (m)')
plt.plot(t_interp, traj_y[0], label='y (m)')
plt.plot(t_interp, traj_theta[0], label='theta (rad)')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('Pose/Orientation')
plt.tight_layout()

plt.figure("Velocity")
plt.title("Velocity", fontsize = 20, fontweight='bold')
plt.plot(t_interp, traj_x[1], label='x (m/s)')
plt.plot(t_interp, traj_y[1], label='y (m/s)')
plt.plot(t_interp, traj_theta[1], label='theta (rad/s)')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('Pose/Orientation')
plt.tight_layout()

plt.figure("Acceleration")
plt.title("Acceleration", fontsize = 20, fontweight='bold')
plt.plot(t_interp, traj_x[2], label='x (m/s^2)')
plt.plot(t_interp, traj_y[2], label='y (m/s^2)')
plt.plot(t_interp, traj_theta[2], label='theta (rad/s^2)')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('Pose/Orientation')
plt.tight_layout()

plt.show()