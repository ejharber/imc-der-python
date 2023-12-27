import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
from scipy.interpolate import PchipInterpolator, CubicSpline, splrep
import imageio
import seaborn as sns

sns.set() # Setting seaborn as default style even if use only matplotlib

allData_train = np.load('alldata_1_50_3_512.npz')["loss_values_train"]
allData_test = np.load('alldata_1_50_3_512.npz')["loss_values_test"]

noForce_train = np.load('noforce_1_50_3_512.npz')["loss_values_train"]
noForce_test = np.load('noforce_1_50_3_512.npz')["loss_values_test"]

noPose_train = np.load('nopose_1_50_3_512.npz')["loss_values_train"]
noPose_test = np.load('nopose_1_50_3_512.npz')["loss_values_test"]

plt.plot(allData_train, label="All Data (train)")
plt.plot(noForce_train, label="Only Position Data (train)")
plt.plot(noPose_train, label="Only Force Data (train)")

plt.plot(allData_test, label="All Data (test)")
plt.plot(noForce_test, label="Only Position Data (test)")
plt.plot(noPose_test, label="Only Force Data (test)")

# plt.plot(force)
plt.xlabel('Iteration')
plt.ylabel('MSE (m)')
plt.title('Convergence of Iterative Policy')
plt.legend()
plt.show()
