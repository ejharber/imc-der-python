#!/usr/bin/env python
 
import cv2
import numpy as np
import os
import glob

import numpy as np

import matplotlib.pyplot as plt
import numpy as np



# Defining the dimensions of checkerboard
CHECKERBOARD = (17, 11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
# Creating vector to store vectors of 3D points for each checkerboard image
camera_obj_points = []
# Creating vector to store vectors of 2D points for each checkerboard image
camera_img_points = []  
# Creating vector to store vectors of 3D points for each checkerboard image
mocap_obj_points = []

def euclidean_transform_3D(A, B):
    '''
        A,B - Nx3 matrix
        return:
            R - 3x3 rotation matrix
            t = 3x1 column vector
    '''
    assert len(A) == len(B)

    # number of points
    N = A.shape[0]; 

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre matrices
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # covariance of datasets
    H = np.transpose(AA) @  BB

    # matrix decomposition on rotation, scaling and rotation matrices
    U, S, Vt = np.linalg.svd(H)

    # resulting rotation
    R = Vt.T @  U.T
    print('R',R)
    #prinyt(Vt)
    print(Vt)
    # handle svd sign problem
    if np.linalg.det(R) < 0:
        print ("sign")
        # thanks to @valeriy.krygin to pointing me on a bug here
        Vt[2,:] *= -1
        R = Vt.T * U.T
        print('new R',R)

    t = -R @ centroid_A.T + centroid_B.T

    return R, t
 
def interpolate_checkerboard_from_mocap(mocap_marker_data):
    x_mean = np.mean(mocap_marker_data[:, 0])
    y_mean = np.mean(mocap_marker_data[:, 1])
    z_mean = np.mean(mocap_marker_data[:, 2])

    # print(x_mean, y_mean, z_mean)
    # print(mocap_marker_data)
    # print(mocap_marker_data[:, 2] < z_mean, mocap_marker_data[:, 1] > y_mean)

    x_1 = mocap_marker_data[np.logical_and(mocap_marker_data[:, 2] < z_mean, mocap_marker_data[:, 1] < y_mean), :] # bottom left
    x_2 = mocap_marker_data[np.logical_and(mocap_marker_data[:, 2] > z_mean, mocap_marker_data[:, 1] < y_mean), :] # top left
    x_3 = mocap_marker_data[np.logical_and(mocap_marker_data[:, 2] < z_mean, mocap_marker_data[:, 1] > y_mean), :] # bottom right

    # print(x_1, x_2, x_3)

    x_unit_mocap = (x_3 - x_1) / np.linalg.norm(x_3 - x_1) * 43 / 1000
    y_unit_mocap = (x_2 - x_1) / np.linalg.norm(x_2 - x_1) * 43 / 1000
    z_unit_mocap = np.cross(x_unit_mocap, y_unit_mocap)
    z_unit_mocap = z_unit_mocap / np.linalg.norm(z_unit_mocap) * 10 / 1000

    mocap_checkerboard_data = []
    for y_i in range(CHECKERBOARD[1], 0, -1): # top to bottom 
        for x_i in range(1, CHECKERBOARD[0] + 1): # left to right
            interp = x_1 + x_unit_mocap * x_i + y_unit_mocap * y_i - z_unit_mocap
            mocap_checkerboard_data.append(interp[0, :])

    mocap_checkerboard_data = np.array(mocap_checkerboard_data)
    return mocap_checkerboard_data

## Calibrate Camera
for file in os.listdir('data'):

    if not file[-4:] == '.npy': continue 
    sample = int(file[:-4])

    print(file)

    img = cv2.imread("data/" + str(sample) + ".png")
    mocap_marker_data = np.load("data/" + str(sample) + ".npy", allow_pickle=True)
    mocap_checkerboard_data = interpolate_checkerboard_from_mocap(mocap_marker_data)

    # Defining the world coordinates for 3D points
    obj_points = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    i = 0
    for y_i in range(CHECKERBOARD[1], 0, -1):
        for x_i in range(1, CHECKERBOARD[0] + 1):
            obj_points[0, i, 0] = x_i * 43 / 1000
            obj_points[0, i, 1] = y_i * 43 / 1000
            i += 1

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:

        camera_obj_points.append(obj_points)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1,-1), criteria)
         
        camera_img_points.append(corners2)

        # print(corners)
 
        mocap_obj_points.append(np.array([mocap_checkerboard_data]))

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # print(np.array(mocap_obj_points).shape, np.array(camera_obj_points).shape)

    cv2.imshow('img',img)
    cv2.waitKey(0)

    # break
 
cv2.destroyAllWindows()

#calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(camera_obj_points, camera_img_points, gray.shape[::-1], None, None)

# rotates and translates the imgpoints into the world frame
img_data = []
for i in range(len(camera_obj_points)):
    R = np.array(cv2.Rodrigues(rvecs[i])[0])
    obj_pts = R @ (camera_obj_points[i][0, :, :].T)
    obj_offset = tvecs[i]
    print(obj_pts.shape, obj_offset.shape)
    
    img_data.append((obj_pts + obj_offset).T)
    # print(np.array(img_data).shape)
img_data = np.array(img_data)

# find the transformation between the mocap data and the img data
mocap_data = np.array(mocap_obj_points)
mocap_data = mocap_data[:, 0, :, :]
img_data = img_data.reshape((int(img_data.shape[0] * img_data.shape[1]), 3))
mocap_data = mocap_data.reshape((int(mocap_data.shape[0] * mocap_data.shape[1]), 3))

# print(img_data.shape)
R, t = euclidean_transform_3D(mocap_data, img_data)
t = np.array([t]).T

# ax = plt.axes(projection='3d')
# print(mocap_data.shape)
# Plot a sin curve using the x and y axes.
# x = np.linspace(0, 1, 100)
# y = np.sin(x * 2 * np.pi) / 2 + 0.5
# data = R @ mocap_data.T + t
# ax.plot(data[0,:], data[1,:], data[2,:], 'r.')
# data = img_data.T
# ax.plot(data[0,:], data[1,:], data[2,:], 'b.')
# plt.show()
# exit()
# R @ mocap_data + t


for file in os.listdir('data'):

    if not file[-4:] == '.npy': continue 
    i = int(file[:-4])

    img = cv2.imread("data/" + str(i) + ".png")
    img = cv2.undistort(img, mtx, dist, None, mtx)

    mocap = np.load("data/" + str(i) + ".npy", allow_pickle=True)

    x_mean = np.mean(mocap[:, 0])
    y_mean = np.mean(mocap[:, 1])
    z_mean = np.mean(mocap[:, 2])

    x_1 = mocap[np.logical_and(mocap[:, 2] < z_mean, mocap[:, 1] < y_mean), :]
    x_2 = mocap[np.logical_and(mocap[:, 2] > z_mean, mocap[:, 1] < y_mean), :]
    x_3 = mocap[np.logical_and(mocap[:, 2] < z_mean, mocap[:, 1] > y_mean), :]

    # print(np.linalg.norm(mocap, axis = 1))

    x_unit = (x_3 - x_1) / np.linalg.norm(x_3 - x_1) * 43 / 1000
    y_unit = (x_2 - x_1) / np.linalg.norm(x_2 - x_1) * 43 / 1000
    z_unit = np.cross(x_unit, y_unit)
    z_unit = z_unit / np.linalg.norm(z_unit) * 10 / 1000

    mocap_data = []

    for y_i in range(CHECKERBOARD[1], 0, -1):
        for x_i in range(1, CHECKERBOARD[0] + 1):
            print(x_i, y_i)
            mocap_data.append(x_unit * x_i + y_unit * y_i + x_1 + z_unit)

    mocap_data = np.array(mocap_data)

    imgpoints2, _ = cv2.projectPoints(mocap_data, R, t, mtx, dist)
    for j in range(CHECKERBOARD[0] * CHECKERBOARD[1]): 
        img = cv2.circle(img, (int(imgpoints2[j, 0, 0]), int(imgpoints2[j, 0, 1])), 3, (0,0,255), -1)

    j = 0
    imgpoints2, _ = cv2.projectPoints(x_1, R, t, mtx, dist)
    img = cv2.circle(img, (int(imgpoints2[j, 0, 0]), int(imgpoints2[j, 0, 1])), 3, (0,0,0), -1)

    imgpoints2, _ = cv2.projectPoints(x_2, R, t, mtx, dist)
    img = cv2.circle(img, (int(imgpoints2[j, 0, 0]), int(imgpoints2[j, 0, 1])), 3, (0,0,255), -1)

    imgpoints2, _ = cv2.projectPoints(x_3, R, t, mtx, dist)
    img = cv2.circle(img, (int(imgpoints2[j, 0, 0]), int(imgpoints2[j, 0, 1])), 3, (255,0,0), -1)

    # print(imgpoints2)
    cv2.imshow('img',img)
    cv2.waitKey(0)
 
cv2.destroyAllWindows()

np.savez("calibration_data", R=R, t=t, mtx=mtx, dist=dist)


