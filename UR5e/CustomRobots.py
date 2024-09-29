from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.robot.DHRobot import DHRobot, DHLink
import roboticstoolbox as rtb
import os 
import numpy as np 
import spatialmath as sm
from math import pi
from scipy.spatial.transform import Rotation as R

import os 

class UR5eCustom(Robot):
    def __init__(self):

        urdf_path = os.path.dirname(os.path.realpath(__file__)) + "/models/"

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "ur5e.urdf",
            tld=urdf_path
        )

        super().__init__(
            links,
            name=name.upper(),
            manufacturer="Universal Robotics",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # self.qr = np.array([np.pi, 0, 0, 0, np.pi / 2, 0])
        self.qr = np.array([np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

    def create_trajectory(self, q0, qf, time=1, dt = 0.002):
        def quintic_func(q0, qf, T, qd0=0, qdf=0):
            X = [
                [ 0.0,          0.0,         0.0,        0.0,     0.0,  1.0],
                [ T**5,         T**4,        T**3,       T**2,    T,    1.0],
                [ 0.0,          0.0,         0.0,        0.0,     1.0,  0.0],
                [ 5.0 * T**4,   4.0 * T**3,  3.0 * T**2, 2.0 * T, 1.0,  0.0],
                [ 0.0,          0.0,         0.0,        2.0,     0.0,  0.0],
                [20.0 * T**3,  12.0 * T**2,  6.0 * T,    2.0,     0.0,  0.0],
            ]
            # fmt: on
            coeffs, resid, rank, s = np.linalg.lstsq(
                X, np.r_[q0, qf, qd0, qdf, 0, 0], rcond=None
            )

            # coefficients of derivatives
            coeffs_d = coeffs[0:5] * np.arange(5, 0, -1)
            coeffs_dd = coeffs_d[0:4] * np.arange(4, 0, -1)

            return lambda x: (
                np.polyval(coeffs, x),
                np.polyval(coeffs_d, x),
                np.polyval(coeffs_dd, x),
            )

        def quintic(q0, qf, t, qd0=0, qdf=0):
            tf = max(t)

            polyfunc = quintic_func(q0, qf, tf, qd0, qdf)

            # evaluate the polynomials
            traj = polyfunc(t)
            p = traj[0]
            pd = traj[1]
            pdd = traj[2]

            return p, pd, pdd

        weighpoints = [[q0[i], qf[i]] for i in range(6)]
        weighpoints = np.array(weighpoints)

        traj = []
        traj_u = []
        traj_a = []

        for dim in range(weighpoints.shape[0]):
            x_1_traj = [0, time]
            x_2_traj = [weighpoints[dim, 0], weighpoints[dim, 1]]
            t = np.linspace(0, time, round(time/dt))
            p, pd, pdd = quintic(x_2_traj[0], x_2_traj[1], t)
            traj.append(p)
            traj_u.append(pd)
            traj_a.append(pdd)

        traj = np.array(traj) * np.pi / 180.0
        traj_u = np.array(traj_u) * np.pi / 180.0
        traj_a = np.array(traj_a) * np.pi / 180.0

        return traj

    def fk_traj(self, traj, two_dimention=False):
        def getAngle(P, Q):
            R = np.dot(P, Q.T)
            cos_theta = (np.trace(R)-1)/2
            return np.arccos(cos_theta)
        traj_fk = []

        zero_pose = None
        for i in range(traj.shape[1]):

            pose = np.array(self.fkine(traj[:, i]))[:3, 3]
            orientation = np.array(self.fkine(traj[:, i]))[:3, :3]

            if two_dimention:
                if zero_pose is None:
                    zero_pose = np.copy(orientation)
                pose = [pose[0], pose[2]]
                orientation = np.array([getAngle(zero_pose, orientation)]) - np.pi/2

            else:
                orientation = R.from_matrix(orientation).as_rotvec()

            fk = np.append(pose, orientation)

            traj_fk.append(fk)

        return np.array(traj_fk)

    def fk_traj_stick(self, traj, two_dimention=False):
        def getAngle(P, Q): # rodriguiz formula
            R = np.dot(P, Q.T)
            cos_theta = (np.trace(R)-1)/2
            if cos_theta >= 1 and cos_theta <= 1 + 1e-7: # floating point error
                cos_theta = 1
            return np.arccos(cos_theta)

        traj_fk = []

        dir_path = os.path.dirname(os.path.realpath(__file__))
        T_robot1_robot2 = np.load(dir_path + "/mocap_calib.npz")["T_robot1_robot2"]

        zero_pose = None
        for i in range(traj.shape[1]):

            out = np.array(self.fkine(traj[:, i])) @ T_robot1_robot2

            pose = out[:3, 3]
            orientation = out[:3, :3]

            if two_dimention:
                if zero_pose is None:
                    zero_pose = np.copy(orientation)
                pose = [pose[0], pose[2]]
                orientation = np.array([getAngle(zero_pose, orientation)]) - np.pi/2

            else:
                orientation = R.from_matrix(orientation).as_rotvec()

            fk = np.append(pose, orientation)

            traj_fk.append(fk)

        traj_fk = np.array(traj_fk)

        if two_dimention:
            I = np.where(np.diff(traj_fk[:, 2]) <= 0)[0]
            if I.shape[0] > 0:
                I = I[0]
                traj_fk[I:, 2] = traj_fk[I:, 2]
            traj_fk[:, 2] = traj_fk[:, 2]

        return traj_fk

    def fk_traj_stick_world(self, traj, mocap_base_to_world):
        def convert_quat_to_matrix(vec):
            T = np.zeros((4, 4))
            T[:3, -1] = vec[:3]
            T[:3, :3] = R.from_quat(vec[[4, 5, 6, 3]]).as_matrix()
            T[3, 3] = 1
            return T

        mocap_base_to_world = convert_quat_to_matrix(mocap_base_to_world)

        traj_fk = []

        dir_path = os.path.dirname(os.path.realpath(__file__))
        T_robot1_robot2 = np.load(dir_path + "/mocap_calib.npz")["T_robot1_robot2"]
        T_base2_base1 = np.load(dir_path + "/mocap_calib.npz")["T_base2_base1"]

        zero_pose = None
        for i in range(traj.shape[1]):

            out = mocap_base_to_world @ T_base2_base1 @ np.array(self.fkine(traj[:, i])) @ T_robot1_robot2

            pose = out[:3, 3]
            orientation = out[:3, :3]
            orientation = R.from_matrix(orientation).as_rotvec()

            fk = np.append(pose, orientation)

            traj_fk.append(fk)

        traj_fk = np.array(traj_fk)

        return traj_fk

    def convert_worktraj_to_robot(self, traj_world, mocap_base_to_world, two_dimention=False):
        def convert_quat_to_matrix(vec):
            if len(vec.shape) == 2:
                T = np.zeros((vec.shape[0], 4, 4))
                for i in range(vec.shape[0]):
                    T[i, :3, -1] = vec[i, :3]
                    T[i, :3, :3] = R.from_quat(vec[i, [4, 5, 6, 3]]).as_matrix()
                    T[i, 3, 3] = 1
                return T
            else:
                T = np.zeros((4, 4))
                T[:3, -1] = vec[:3]
                T[:3, :3] = R.from_quat(vec[[4, 5, 6, 3]]).as_matrix()
                T[3, 3] = 1
                return T

        def convert_matrix_to_quat(matrix):
            T = np.zeros((matrix.shape[0], 7))
            for i in range(matrix.shape[0]):
                T[i, :3] = matrix[i, :3, -1]
                T[i, 3:] = R.from_matrix(matrix[i, :3, :3]).as_quat()[[1, 2, 3, 0]]
            return T

        mocap_base_to_world = convert_quat_to_matrix(mocap_base_to_world)
        traj_world = convert_quat_to_matrix(traj_world)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        T_base2_base1 = np.load(dir_path + "/mocap_calib.npz")["T_base2_base1"]

        for i in range(traj_world.shape[0]):
            traj_world[i, :, :] = np.linalg.inv(mocap_base_to_world @ T_base2_base1) @ traj_world[i, :, :]

        out = convert_matrix_to_quat(traj_world)

        return out

    def convert_workpoint_to_robot(self, point_world, mocap_base_to_world, two_dimention=False):
        def convert_quat_to_matrix(vec):
            if len(vec.shape) == 2:
                T = np.zeros((vec.shape[0], 4, 4))
                for i in range(vec.shape[0]):
                    T[i, :3, -1] = vec[i, :3]
                    T[i, :3, :3] = R.from_quat(vec[i, [4, 5, 6, 3]]).as_matrix()
                    T[i, 3, 3] = 1
                return T
            else:
                T = np.zeros((4, 4))
                T[:3, -1] = vec[:3]
                T[:3, :3] = R.from_quat(vec[[4, 5, 6, 3]]).as_matrix()
                T[3, 3] = 1
                return T

        def convert_matrix_to_quat(matrix):
            T = np.zeros((matrix.shape[0], 7))
            for i in range(matrix.shape[0]):
                T[i, :3] = matrix[i, :3, -1]
                T[i, 3:] = R.from_matrix(matrix[i, :3, :3]).as_quat()[[1, 2, 3, 0]]
            return T

        mocap_base_to_world = convert_quat_to_matrix(mocap_base_to_world)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        T_base2_base1 = np.load(dir_path + "/mocap_calib.npz")["T_base2_base1"]

        point_world = [point_world[0], point_world[1], point_world[2], 0]
        out = np.linalg.inv(mocap_base_to_world @ T_base2_base1) @ point_world

        print(out)
        return out[[0, 2]]

    def convert_robotpoint_to_world(self, goal, mocap_base_to_world):
        def convert_quat_to_matrix(vec):
            if len(vec.shape) == 2:
                T = np.zeros((vec.shape[0], 4, 4))
                for i in range(vec.shape[0]):
                    T[i, :3, -1] = vec[i, :3]
                    T[i, :3, :3] = R.from_quat(vec[i, [4, 5, 6, 3]]).as_matrix()
                    T[i, 3, 3] = 1
                return T
            else:
                T = np.zeros((4, 4))
                T[:3, -1] = vec[:3]
                T[:3, :3] = R.from_quat(vec[[4, 5, 6, 3]]).as_matrix()
                T[3, 3] = 1
                return T

        goal = np.array([goal[0], 0.1307183, goal[1], 1])

        mocap_base_to_world = convert_quat_to_matrix(mocap_base_to_world)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        T_base2_base1 = np.load(dir_path + "/mocap_calib.npz")["T_base2_base1"]

        out = mocap_base_to_world @ T_base2_base1 @ goal

        return out[:3]


def main():
    # some test code
    robot_path = os.path.join(os.getcwd(), 'models')
    UR5e = UR5eCustom(robot_path)

    q0 = [0, -54, 134, -167, -90, 0]
    qf = [0, -90 - 5, 95 - 5, -180 - 10, -90, 0]

    traj = UR5e.create_trajectory(q0, qf)

if __name__ == '__main__':
    main()












