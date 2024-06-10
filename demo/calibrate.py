#!/usr/bin/env python
 
import cv2
import numpy as np
import os
import glob

import numpy as np

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
 
# Defining the dimensions of checkerboard
CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []  

mocap_data = []
img_data = []

for file in os.listdir('data'):

    if not file[-4:] == '.npy': continue 
    i = int(file[:-4])

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * 22.5

    img = cv2.imread("data/" + str(i) + ".png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:

        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1,-1), criteria)
         
        imgpoints.append(corners2)
 
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        mocap = np.load("data/" + str(i) + ".png.npy", allow_pickle=True)

        x_mean = np.mean(mocap[:, 0])
        y_mean = np.mean(mocap[:, 1])
        z_mean = np.mean(mocap[:, 2])

        x_1 = mocap[np.logical_and(mocap[:, 2] < z_mean, mocap[:, 1] < y_mean), :]
        x_2 = mocap[np.logical_and(mocap[:, 2] > z_mean, mocap[:, 1] < y_mean), :]
        x_3 = mocap[np.logical_and(mocap[:, 2] < z_mean, mocap[:, 1] > y_mean), :]

        x_unit = (x_3 - x_1) / np.linalg.norm(x_3 - x_1) * 22.5
        y_unit = (x_2 - x_1) / np.linalg.norm(x_2 - x_1) * 22.5

        z_unit = np.cross(x_unit, y_unit)
        z_unit = z_unit / np.linalg.norm(z_unit) * 10

        for y_i in range(CHECKERBOARD[1], 0, -1):
            for x_i in range(1, CHECKERBOARD[0] + 1):
                # print(x_i, y_i)
                mocap_data.append(x_unit * x_i + y_unit * y_i + x_1 - z_unit)

        break

    # cv2.imshow('img',img)
    # cv2.waitKey(0)
 
# cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# rotates and translates the imgpoints into the world frame
for i in range(len(objpoints)):
    R = np.array(cv2.Rodrigues(rvecs[i])[0])
    obj_pts = R @ (objpoints[i][0, :, :].T)
    obj_offset = tvecs[i]
    print(obj_pts.shape, obj_offset.shape)
    # img_data.append((np.linalg.inv(R) @ (objpoints[i][0, :, :].T - tvecs[i])))
    # print(objpoints[i][0, :, :].shape)
    # print(R.shape)
    # print((R @ objpoints[i][0, :, :]).shape)
    # print(tvecs[i].shape)
    img_data.append(obj_pts + obj_offset)
    # print(np.array(img_data).shape)

# find the transformation between the mocap data and the img data
mocap_data = np.array(mocap_data)
mocap_data = mocap_data[:, 0, :]
img_data = np.array(img_data)
print(img_data.shape, (int(img_data.shape[0] * img_data.shape[1])))
img_data = img_data.reshape((int(img_data.shape[0] * img_data.shape[1]), 3))
# print(img_data.shape)
R, t = euclidean_transform_3D(mocap_data, img_data)
t = np.array([t]).T

import matplotlib.pyplot as plt
import numpy as np

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
    i = int(file[:-8])

    img = cv2.imread("data/" + str(i) + ".png")

    mocap = np.load("data/" + str(i) + ".png.npy", allow_pickle=True)

    x_mean = np.mean(mocap[:, 0])
    y_mean = np.mean(mocap[:, 1])
    z_mean = np.mean(mocap[:, 2])

    x_1 = mocap[np.logical_and(mocap[:, 2] < z_mean, mocap[:, 1] < y_mean), :]
    x_2 = mocap[np.logical_and(mocap[:, 2] > z_mean, mocap[:, 1] < y_mean), :]
    x_3 = mocap[np.logical_and(mocap[:, 2] < z_mean, mocap[:, 1] > y_mean), :]


    # print(np.linalg.norm(mocap, axis = 1))

    x_unit = (x_3 - x_1) / np.linalg.norm(x_3 - x_1) * 22.5
    y_unit = (x_2 - x_1) / np.linalg.norm(x_2 - x_1) * 22.5
    z_unit = np.cross(x_unit, y_unit)
    z_unit = z_unit / np.linalg.norm(z_unit) * 10

    mocap_data = []

    for y_i in range(CHECKERBOARD[1], 0, -1):
        for x_i in range(1, CHECKERBOARD[0] + 1):
            print(x_i, y_i)
            mocap_data.append(x_unit * x_i + y_unit * y_i + x_1 - z_unit)

    mocap_data = np.array(mocap_data)

    # Find the chess board corners
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # print(ret)
    # if ret == True:

        # corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1,-1), criteria)

    # imgpoints2, _ = cv2.projectPoints(objpoints, R, t, mtx, dist)
    # for j in range(48): 
        # img = cv2.circle(img, (int(imgpoints2[j, 0, 0]), int(imgpoints2[j, 0, 1])), 3, (255,255,255), -1)

    imgpoints2, _ = cv2.projectPoints(mocap_data, R, t, mtx, dist)
    for j in range(48): 
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


