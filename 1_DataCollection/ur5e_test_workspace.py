#!/usr/bin/env python3

import rtde_control
import numpy as np

# ros packages
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState

import numpy as np
import time

def create_trajectory(weighpoints):
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

        return p

    traj = []
    time = 5 # set sim time to be 0.5 seconds

    for dim in range(weighpoints.shape[0]):
        x_1_traj = [0, time]
        x_2_traj = [weighpoints[dim, 0], weighpoints[dim, 1]]
        t = np.linspace(0, time, round(time*500)) # can send data at 500 hz
        traj.append(quintic(x_2_traj[0], x_2_traj[1], t))

    traj = np.array(traj)

    return traj


class UR5E(object):
    def __init__(self):

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.10.60")

        self.home_joint_pose = [180, -110, -140, -30, 90, 0]

        time.sleep(1)

        self.go_to_home()

    def go_to_home(self):
        print('go home')
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q)

    def map_work_space_servoj(self):
        print('start mapping')

        for q1 in np.linspace(-110, -60, 5):
            for q2 in np.linspace(-140, -80, 5):
                for q3 in np.linspace(-30, -30 + 90, 5):
                    self.go_to_home()

                    q = [180, q1 ,q2, q3, 90, 0]
                    # q = [180, -90, -100, 0, 90, 0]

                    weighpoints = [[self.home_joint_pose[i], q[i]] for i in range(6)]
                    weighpoints = np.array(weighpoints)
                    traj = create_trajectory(weighpoints)

                    # Parameters
                    velocity = 3
                    acceleration = 5
                    dt = 1.0/500  # 2ms
                    lookahead_time = 0.05
                    gain = 1000

                    for i in range(traj.shape[1]):
                        q = traj[:, i]

                        q = q * np.pi / 180
                        t_start = self.rtde_c.initPeriod()
                        self.rtde_c.servoJ(q, velocity, acceleration, dt, lookahead_time, gain)
                        self.rtde_c.waitPeriod(t_start)

                    self.rtde_c.servoStop()

        self.go_to_home()

    def map_work_space_movej(self):
        print('start mapping')

        for q1 in np.linspace(-110, -60, 5):
            for q2 in np.linspace(-140, -80, 5):
                for q3 in np.linspace(-30, -30 + 90, 5):
                    self.go_to_home()
                    q = [180, q1 ,q2, q3, 90, 0]
                    q = np.array(q) * np.pi / 180.0
                    self.rtde_c.moveJ(q)

        self.go_to_home()


    def reset_rope(self):
        q = np.copy(self.home_joint_pose)
        q[1] -= 20
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q, speed = 0.05, acceleration = 0.1)

        time.sleep(2)
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q, speed = 0.05, acceleration = 0.1)
        time.sleep(2)

    def force_sensor_experiment(self):

        print('start mapping')

        for trial in range(10):
            self.go_to_home()
            self.reset_rope()

            q = [180, -90 ,-90, 0, 90, 0]
            q = np.array(q) * np.pi / 180.0
            self.rtde_c.moveJ(q, speed = 2, acceleration = 4)
            self.go_to_home()

    # def __del__(self):
    #     if self.rtde_c is not None:
    #         self.rtde_c.stopScript()

if __name__ == '__main__':
    ur5e = UR5E()
    ur5e.map_work_space_servoj()

