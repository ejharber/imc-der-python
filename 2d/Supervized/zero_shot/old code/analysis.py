import numpy as np 
import matplotlib.pyplot as plt


gt = np.load("analysis/groundtruth.npy")
nn = np.load("analysis/128_128.npy")
nn2 = np.load("analysis/64_64.npy")

x = [gt.flatten(), nn, nn2]
x = np.array(x)

plt.boxplot(x)
plt.show()