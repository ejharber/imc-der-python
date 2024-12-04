import numpy as np

# Values to be saved
values = np.array([
    3.15919883e-04, 3.68846405e-02, 7.48713860e-01, 6.42394634e-02,
    1.05878645e+00, 1.90501696e+01, 1.00000000e+01, 1.10283528e-04,
    3.13799413e-03, 8.49136498e-05, 1.00000000e-01, 2.03000000e-02,
    7.88000000e-02, 5.01003271e+02, 5.16423369e+02
])

# Save as an npz file
np.savez('params/N2.npz', params=values)

print("Values saved to 'params.npz'")
