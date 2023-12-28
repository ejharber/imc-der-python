import matplotlib.pyplot as plt
import numpy as np
import os 

X = np.array([])
y = np.array([])

for file in os.listdir("data"):
    print(file)
    try:
        data = np.load("data/" + file, allow_pickle=True)
    except:
        continue 

    # print(X.shape)
    if len(X.shape) == 1:
        X = data["actions"]
        y = np.array(data["trajs_pos"])[:,-2:,-1]

    else:
        X = np.append(X, data["actions"], axis=0)
        y = np.append(y, data["trajs_pos"][:,-2:,-1], axis=0)

    if X.shape[0] > 1_000_000:
        break

success = []
failure = []

print(X.shape, y.shape)

for i in range(y.shape[0]):
    if not np.all(y[i, :] == 0):
        success.append(i)
    else:
        failure.append(i)

# X_success = X[success, :]
# y_success = y[success, :]

plt.figure()
plt.plot(X[success, 0], X[success, 1], 'b.')
plt.plot(X[failure, 0], X[failure, 1], 'r.')
# plt.show()

plt.figure()
plt.plot(X[success, 2], 'b.')
plt.plot(X[failure, 2], 'r.')
plt.show()
