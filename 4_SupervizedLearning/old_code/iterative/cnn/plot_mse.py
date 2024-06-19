import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
from scipy.interpolate import PchipInterpolator, CubicSpline, splrep
import imageio
import seaborn as sns

sns.set() # Setting seaborn as default style even if use only matplotlib


mse = np.load('mse.npy')

no_force = mse[1, :]
force = mse[2, :]
# print(no_force.shape)
# print(mse.shape)
plt.plot(no_force)
plt.plot(force)
plt.xlabel('Iteration')
plt.ylabel('MSE (m)')
plt.title('Convergence of Iterative Policy')
plt.show()
