#!/usr/bin/env python

import cv2
import numpy as np
import os

# Define checkerboard dimensions
CHECKERBOARD = (17, 11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

# Initialize lists to store 3D and 2D points
camera_obj_points, camera_img_points, mocap_obj_points = [], [], []

def euclidean_transform_3D(A, B):
    """Compute the Euclidean transformation between two sets of 3D points."""
    assert len(A) == len(B)
    N = A.shape[0]
    centroid_A, centroid_B = np.mean(A, axis=0), np.mean(B, axis=0)
    H = ((A - centroid_A).T @ (B - centroid_B))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A.T + centroid_B.T
    return R, t

def interpolate_checkerboard_from_mocap(mocap_marker_data):
    """Interpolates 3D checkerboard points from mocap data points."""
    x_mean, y_mean, z_mean = np.mean(mocap_marker_data, axis=0)
    x_1 = mocap_marker_data[np.logical_and(mocap_marker_data[:, 2] < z_mean, mocap_marker_data[:, 1] < y_mean)]
    x_2 = mocap_marker_data[np.logical_and(mocap_marker_data[:, 2] > z_mean, mocap_marker_data[:, 1] < y_mean)]
    x_3 = mocap_marker_data[np.logical_and(mocap_marker_data[:, 2] < z_mean, mocap_marker_data[:, 1] > y_mean)]

    x_unit = (x_3 - x_1) / np.linalg.norm(x_3 - x_1) * 0.043
    y_unit = (x_2 - x_1) / np.linalg.norm(x_2 - x_1) * 0.043
    z_unit = np.cross(x_unit, y_unit) / np.linalg.norm(np.cross(x_unit, y_unit)) * 0.01

    mocap_checkerboard_data = [(x_unit * x_i + y_unit * y_i + x_1 + z_unit)[0]
                               for y_i in range(CHECKERBOARD[1], 0, -1)
                               for x_i in range(1, CHECKERBOARD[0] + 1)]
    return np.array(mocap_checkerboard_data)

# Prompt for data folder
data_folder = input("Enter the path to the data folder: ")

# Process each file in the specified folder
for file in os.listdir(data_folder):
    if not file.endswith('.npy'): continue
    sample = int(file[:-4])
    print(f"Processing {file}...")

    img = cv2.imread(os.path.join(data_folder, f"{sample}.png"))
    mocap_marker_data = np.load(os.path.join(data_folder, file), allow_pickle=True)
    mocap_checkerboard_data = interpolate_checkerboard_from_mocap(mocap_marker_data)

    # Define 3D object points for checkerboard
    obj_points = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    obj_points[0, :, :2] = np.mgrid[1:CHECKERBOARD[0] + 1, CHECKERBOARD[1]:0:-1].T.reshape(-1, 2) * 0.043

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        camera_obj_points.append(obj_points)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        # Compute the average y-coordinate of the points
        avg_y = np.mean(corners2[:, 0, 1])

        # Check if the first point is above or below the average
        if corners2[0, 0, 1] > avg_y:
            # The first point is above the average, so we flip the corners to go top to bottom
            corners2 = corners2[::-1]

        camera_img_points.append(corners2)
        mocap_obj_points.append([mocap_checkerboard_data])

        # Draw and display the corners
        # img_with_corners = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('Checkerboard', img_with_corners)
        # cv2.waitKey(0)

cv2.destroyAllWindows()

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(camera_obj_points, camera_img_points, gray.shape[::-1], None, None)

# Transform camera points to mocap points
img_data = []
for i, obj_points in enumerate(camera_obj_points):
    R = np.array(cv2.Rodrigues(rvecs[i])[0])
    transformed_points = R @ obj_points[0].T + tvecs[i].reshape(-1, 1)
    img_data.append(transformed_points.T)
img_data = np.vstack(img_data)

# Compute transformation from mocap to image data
mocap_data = np.vstack(np.vstack(mocap_obj_points))
print(mocap_data.shape, img_data.shape,np.array(mocap_obj_points).shape)
R, t = euclidean_transform_3D(mocap_data, img_data)
t = t.reshape(3, 1)

# Save calibration data
np.savez("calibration_data", R=R, t=t, mtx=mtx, dist=dist)

# Projection of mocap points on images
for file in os.listdir(data_folder):
    if not file.endswith('.npy'): continue
    sample = int(file[:-4])
    img = cv2.imread(os.path.join(data_folder, f"{sample}.png"))
    img = cv2.undistort(img, mtx, dist, None, mtx)

    mocap_marker_data = np.load(os.path.join(data_folder, file), allow_pickle=True)
    projected_points = interpolate_checkerboard_from_mocap(mocap_marker_data)

    # Project points onto image
    imgpoints2, _ = cv2.projectPoints(projected_points, R, t, mtx, dist)
    for point in imgpoints2:
        img = cv2.circle(img, tuple(point[0].astype(int)), 3, (0, 0, 255), -1)

    cv2.imshow('Projected Points', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
