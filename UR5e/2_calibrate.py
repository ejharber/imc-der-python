# import rosbag2_py
# from rosbag2_py import SequentialReader
# from rosidl_runtime_py./n import deserialize_message
import numpy as np 
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize 
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

import sys
from scipy.optimize import differential_evolution

sys.path.append("..")

from CustomRobots import *

UR5e = UR5eCustom()

data = np.load("raw_data/calibration_data.npz")

ur5e_joint_data = data["ur5e_joint_data"]

ur5e_tool_poses = UR5e.fk_traj(ur5e_joint_data.T)
# print(ur5e_tool_poses.shape)
# ur5e_tool_poses_fk = ur5e_tool_poses_fk[:, :3, 3]
# print(ur5e_tool_poses_fk.shape)
# print(ur5e_tool_poses)
# ur5e_tool_poses = data["ur5e_ee_data"]
mocap_tool_poses = data["mocap_ee_data"]
mocap_base_poses = data["mocap_base_data"]

# print(ur5e_tool_poses.shape, mocap_tool_poses.shape, mocap_base_poses.shape)

## BEGIN CALIBRATION

fig = plt.figure("Before")
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = mocap_tool_poses[:, 0]
yline = mocap_tool_poses[:, 1]
zline = mocap_tool_poses[:, 2]
ax.plot3D(xline, yline, zline)

xline = ur5e_tool_poses[:, 0]
yline = ur5e_tool_poses[:, 1]
zline = ur5e_tool_poses[:, 2]
ax.plot3D(xline, yline, zline)

plt.show()

def convert_quat_to_matrix(vec, switch):
    T = np.zeros((vec.shape[0], 4, 4))
    for i in range(vec.shape[0]):
        T[i, :3, -1] = vec[i, :3]
        if switch:
            T[i, :3, :3] = R.from_quat(vec[i, [3, 4, 5, 6]]).as_matrix()
        else:
            T[i, :3, :3] = R.from_quat(vec[i, [4, 5, 6, 3]]).as_matrix()
        T[i, 3, 3] = 1
    return T

def convert_axangle_to_matrix(vec):
    T = np.zeros((vec.shape[0], 4, 4))
    for i in range(vec.shape[0]):
        T[i, :3, -1] = vec[i, :3]
        T[i, :3, :3] = R.from_rotvec(vec[i, [3, 4, 5]]).as_matrix()
        T[i, 3, 3] = 1
    return T

T_mocap_robot2 = convert_quat_to_matrix(mocap_tool_poses, False)
T_mocap_base2 = convert_quat_to_matrix(mocap_base_poses, False)
T_base1_robot1 = convert_axangle_to_matrix(ur5e_tool_poses)

# code for optimization approach
def cost_fun(x, T_mocap_robot2, T_mocap_base2, T_base1_robot1):

    T_base2_base1 = np.eye(4)
    T_base2_base1[:3, -1] = x[:3]
    T_base2_base1[:3, :3] = R.from_euler('xyz', x[3:6]).as_matrix()
    T_base2_base1 = np.expand_dims(T_base2_base1, axis=0)
    T_base2_base1 = np.repeat(T_base2_base1, T_mocap_robot2.shape[0], axis=0)

    T_robot1_robot2 = np.eye(4)
    T_robot1_robot2[:3, -1] = x[6:9]
    T_robot1_robot2[:3, :3] = R.from_euler('xyz', x[9:]).as_matrix()
    T_robot1_robot2 = np.expand_dims(T_robot1_robot2, axis=0)
    T_robot1_robot2 = np.repeat(T_robot1_robot2, T_mocap_robot2.shape[0], axis=0)

    # cost function from mocap frame to mocap end effector 
    cost_1 = T_mocap_robot2 - T_mocap_base2 @ T_base2_base1 @ T_base1_robot1 @ T_robot1_robot2
    
    T_robot2_robot1 = np.eye(4)
    T_robot2_robot1[:3, -1] = x[6:9]
    T_robot2_robot1[:3, :3] = R.from_euler('xyz', x[9:]).as_matrix()
    T_robot2_robot1 = np.linalg.inv(T_robot2_robot1)
    T_robot2_robot1 = np.expand_dims(T_robot2_robot1, axis=0)
    T_robot2_robot1 = np.repeat(T_robot2_robot1, T_mocap_robot2.shape[0], axis=0)

    # cost function from mocap to ur5e end effector 
    cost_2 = T_mocap_robot2 @ T_robot2_robot1 - T_mocap_base2 @ T_base2_base1 @ T_base1_robot1

    # print("cost1", round(np.mean(np.linalg.norm(cost_1[:, :3, -1], axis=1)), 4), round(np.std(np.linalg.norm(cost_1[:, :3, -1], axis=1)), 4))    
    # print("cost2", round(np.mean(np.linalg.norm(cost_2[:, :3, -1], axis=1)), 4), round(np.std(np.linalg.norm(cost_2[:, :3, -1], axis=1)), 4))

    cost = np.linalg.norm(cost_1[:, :3, -1]) + np.linalg.norm(cost_2[:, :3, -1])

    return cost

min_fun = np.inf
min_x = None

for _ in range(20):
    x0 = np.random.rand(12)
    bounds = [(-2, 2), (-2, 2), (-2, 2), 
              (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi), 
              (-2, 2), (-2, 2), (-2, 2), 
              (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi), (-2*np.pi, 2*np.pi)]

    for i in range(x0.shape[0]):
        low, high = bounds[i]
        x0[i] = x0[i] * (high - low) + low

    fun = lambda x: cost_fun(x, T_mocap_robot2, T_mocap_base2, T_base1_robot1)
    x = minimize(fun, x0, bounds=bounds)
    # x = differential_evolution(cost_fun,                  # the function to minimize
    #                          bounds=bounds,
    #                          args = (T_mocap_robot2, T_mocap_base2, T_base1_robot1),
    #                          tol=0.0001,
    #                          # maxiter=1000,
    #                          workers=-1,
    #                          updating="deferred",
    #                          disp=True)   # the random seed

    # print(x)
    if x.fun < min_fun:
        min_fun = x.fun
        min_x = x.x
        print(min_fun)

    # cost_2 = cost_fun(x.x, T_mocap_robot2, T_mocap_base2, T_base1_robot1)
    # print(cost_2)

x = min_x
cost_fun(x, T_mocap_robot2, T_mocap_base2, T_base1_robot1)

T_base2_base1 = np.eye(4)
T_base2_base1[:3, -1] = x[:3]
T_base2_base1[:3, :3] = R.from_euler('xyz', x[3:6]).as_matrix()

T_base1_base2 = np.linalg.inv(T_base2_base1)

T_robot1_robot2 = np.eye(4)
T_robot1_robot2[:3, -1] = x[6:9]
T_robot1_robot2[:3, :3] = R.from_euler('xyz', x[9:]).as_matrix()

T_robot2_robot1 = np.linalg.inv(T_robot1_robot2)

cal = T_mocap_base2 @ T_base2_base1 @ T_base1_robot1 @ T_robot1_robot2

fig = plt.figure("After")
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = T_mocap_robot2[:, 0, -1]
yline = T_mocap_robot2[:, 1, -1]
zline = T_mocap_robot2[:, 2, -1]
ax.plot3D(xline, yline, zline)

xline = cal[:, 0, -1]
yline = cal[:, 1, -1]
zline = cal[:, 2, -1]
ax.plot3D(xline, yline, zline)

cal_1 = T_mocap_robot2 @ T_robot2_robot1
cal_2 = T_mocap_base2 @ T_base2_base1 @ T_base1_robot1 

fig = plt.figure("After RobotEE")
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = cal_1[:, 0, -1]
yline = cal_1[:, 1, -1]
zline = cal_1[:, 2, -1]
ax.plot3D(xline, yline, zline, 'r.')
ax.plot3D(xline, yline, zline, 'r-')

xline = cal_2[:, 0, -1]
yline = cal_2[:, 1, -1]
zline = cal_2[:, 2, -1]
ax.plot3D(xline, yline, zline, 'b.')
ax.plot3D(xline, yline, zline, 'b-')

plt.show()

print(min_x)

np.savez("mocap_calib", T_base1_base2=T_base1_base2, T_base2_base1=T_base2_base1, T_robot1_robot2=T_robot1_robot2, T_robot2_robot1=T_robot2_robot1)









