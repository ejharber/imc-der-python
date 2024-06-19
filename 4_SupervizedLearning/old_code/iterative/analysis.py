import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib.animation as animation
from scipy.interpolate import PchipInterpolator, CubicSpline, splrep
import imageio
import seaborn as sns

import sys
file_name = 'models_test/LSTM_'
sys.path.append("models_daction_dgoal_full")

sns.set() # Setting seaborn as default style even if use only matplotlib

allData_train = np.load(file_name + 'alldata_2_20_3_500.npz')["loss_values_train"]
allData_test = np.load(file_name + 'alldata_2_20_3_500.npz')["loss_values_test"]

noForce_train = np.load(file_name + 'noforce_2_20_3_500.npz')["loss_values_train"]
noForce_test = np.load(file_name + 'noforce_2_20_3_500.npz')["loss_values_test"]

noPose_train = np.load(file_name + 'nopose_2_20_3_500.npz')["loss_values_train"]
noPose_test = np.load(file_name + 'nopose_2_20_3_500.npz')["loss_values_test"]

plt.figure()
plt.plot(np.log10(allData_train), label="All Data (train)")
plt.plot(np.log10(noForce_train), label="Only Position Data (train)")
plt.plot(np.log10(noPose_train), label="Only Force Data (train)")

plt.xlabel('Mini Batch Iteration')
plt.ylabel('log(MSE (m))')
plt.title('Convergence of Iterative Policy')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.log10(allData_train[300:]), label="All Data (train)")
plt.plot(np.log10(noForce_train[300:]), label="Only Position Data (train)")
plt.plot(np.log10(noPose_train[300:]), label="Only Force Data (train)")

plt.xlabel('Mini Batch Iteration')
plt.ylabel('log(MSE (m))')
plt.title('Convergence of Iterative Policy')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.log10(allData_test), label="All Data (test)")
plt.plot(np.log10(noForce_test), label="Only Position Data (test)")
plt.plot(np.log10(noPose_test), label="Only Force Data (test)")

plt.xlabel('Iteration')
plt.ylabel('log(MSE (m))')
plt.title('Convergence of Iterative Policy')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.log10(allData_test[30:]), label="All Data (test)")
plt.plot(np.log10(noForce_test[30:]), label="Only Position Data (test)")
plt.plot(np.log10(noPose_test[30:]), label="Only Force Data (test)")

# plt.plot(force)
plt.xlabel('Iteration')
plt.ylabel('log(MSE (m))')
plt.title('Convergence of Iterative Policy')
plt.legend()
plt.show()
